import ctypes
import gc
import hashlib
import json
import math
import os
import resource
import shutil
import subprocess
import threading
import time
import warnings
import statistics
from collections import OrderedDict, deque, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import torch
import numpy as np

from .archive import ThinArchive
from .execution_plan import compile_execution_plan
from .operator_registry import select_runtime_executor
from .model_arch import descriptor_from_manifest
from .page_policy import is_body_weight, is_expert_weight, page_role
from .torch_loader import load_tensor_view, map_dtype


def malloc_trim() -> None:
    if os.name != "posix":
        return
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        pass


def quantize_to_fp8_via_gpu(
    source_cpu: torch.Tensor,
    device: torch.device = torch.device("cuda"),
    scale_block_size: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    source_gpu = source_cpu.to(device=device, dtype=torch.bfloat16)
    rows, cols = source_gpu.shape
    scale_blocks = (
        (cols + scale_block_size - 1) // scale_block_size
        if scale_block_size > 0
        else 1
    )
    scales_gpu = torch.empty(
        (rows, scale_blocks) if scale_block_size > 0 else (rows,),
        dtype=torch.float32,
        device=device,
    )
    quantized_bf16 = torch.empty((rows, cols), dtype=torch.bfloat16, device=device)
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    chunk_rows = 4096
    for start in range(0, rows, chunk_rows):
        end = min(rows, start + chunk_rows)
        chunk = source_gpu[start:end]
        if scale_block_size > 0:
            for block in range(scale_blocks):
                col_start = block * scale_block_size
                col_end = min(cols, col_start + scale_block_size)
                block_values = chunk[:, col_start:col_end]
                scale_chunk = (
                    block_values.abs()
                    .amax(dim=1)
                    .float()
                    .clamp_min_(1e-12)
                    .div_(fp8_max)
                )
                scales_gpu[start:end, block].copy_(scale_chunk)
                quantized_bf16[start:end, col_start:col_end].copy_(
                    (block_values.float() / scale_chunk[:, None]).to(dtype=torch.bfloat16)
                )
        else:
            scale_chunk = chunk.abs().amax(dim=1).float().clamp_min_(1e-12).div_(fp8_max)
            scales_gpu[start:end].copy_(scale_chunk)
            quantized_bf16[start:end].copy_(
                (chunk.float() / scale_chunk[:, None]).to(dtype=torch.bfloat16)
            )
    quantized_gpu = quantized_bf16.to(dtype=torch.float8_e4m3fn)
    quantized_cpu = quantized_gpu.to(device="cpu")
    scales_cpu = scales_gpu.to(device="cpu")
    return quantized_cpu, scales_cpu


def quantize_to_int4_cpu(
    source: torch.Tensor,
    *,
    group_size: int = 128,
    chunk_rows: int = 256,
    input_rms: Optional[torch.Tensor] = None,
    input_covariance32: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack a matrix into grouped INT4 without retaining BF16 RAM."""
    rows, cols = int(source.shape[0]), int(source.shape[1])
    if cols % 2:
        raise ValueError("expert INT4 requires an even input width")
    if group_size <= 0 or group_size & (group_size - 1):
        raise ValueError("expert INT4 group size must be a positive power of two")
    groups = (cols + group_size - 1) // group_size
    packed = torch.empty((rows, cols // 2), dtype=torch.uint8)
    # FP16 retains substantially more scale precision than the BF16 source
    # needs while halving scale residency/traffic versus the old FP32 layout.
    # Triton accumulation still promotes loaded scales to FP32.
    # Keep the vocabulary head on its separately planned symmetric format;
    # affine mode targets decoder projections and its doubled metadata is
    # accounted by their PagePool specs below.
    affine = (
        os.environ.get("THINTENSOR_AFFINE_INT4", "0") == "1"
        and rows < 65536
    )
    importance = None
    if input_rms is not None:
        if tuple(input_rms.shape) != (cols,):
            raise ValueError(
                f"INT4 input RMS shape {tuple(input_rms.shape)} does not match "
                f"the projection width {(cols,)}"
            )
        # The diagonal-Hessian approximation for E[(Wx-Wq x)^2] weights
        # each weight-column error by E[x^2]. Keep a small floor so a
        # calibration trace cannot make any channel completely irrelevant.
        importance = input_rms.float().square().clamp_min_(1e-8)
    gptq_inverse_cholesky = None
    if input_covariance32 is not None:
        if group_size != 32:
            raise ValueError("GPTQ covariance blocks currently require INT4 group size 32")
        if tuple(input_covariance32.shape) != (groups, 32, 32):
            raise ValueError(
                f"INT4 covariance shape {tuple(input_covariance32.shape)} does not "
                f"match {(groups, 32, 32)}"
            )
        hessian = input_covariance32.float().clone()
        diagonal_mean = hessian.diagonal(dim1=1, dim2=2).mean(dim=1)
        damp_percent = float(os.environ.get("THINTENSOR_GPTQ_DAMP_PERCENT", "0.01"))
        if damp_percent <= 0:
            raise ValueError("THINTENSOR_GPTQ_DAMP_PERCENT must be positive")
        damp = diagonal_mean.mul(damp_percent).clamp_min_(1e-6)
        hessian.diagonal(dim1=1, dim2=2).add_(damp[:, None])
        gptq_inverse_cholesky = torch.linalg.cholesky(
            torch.linalg.inv(hessian), upper=True
        )
    scales = torch.empty(
        (rows, groups, 2) if affine else (rows, groups),
        dtype=torch.float16,
    )
    for start in range(0, rows, chunk_rows):
        end = min(rows, start + chunk_rows)
        values_f32 = source[start:end].float()
        if cols % group_size == 0:
            grouped = values_f32.reshape(end - start, groups, group_size)
            if affine:
                minimum = grouped.amin(dim=2)
                maximum = grouped.amax(dim=2)
                scale = maximum.sub(minimum).clamp_min_(1e-12).div_(15.0)
                quantized_f32 = torch.round(
                    (grouped - minimum[:, :, None]) / scale[:, :, None]
                ).clamp_(0, 15)
                # Refine the affine line against the assigned codes. With a
                # calibration RMS vector this minimizes the diagonal-Hessian
                # approximation to projection-output error, while preserving
                # the exact same packed weights and decode kernel layout.
                group_importance = (
                    importance.reshape(1, groups, group_size)
                    if importance is not None
                    else None
                )
                for _ in range(2):
                    if group_importance is None:
                        q_mean = quantized_f32.mean(dim=2)
                        x_mean = grouped.mean(dim=2)
                        regression_weight = 1.0
                    else:
                        weight_sum = group_importance.sum(dim=2)
                        q_mean = (
                            quantized_f32 * group_importance
                        ).sum(dim=2) / weight_sum
                        x_mean = (grouped * group_importance).sum(dim=2) / weight_sum
                        regression_weight = group_importance
                    q_centered = quantized_f32 - q_mean[:, :, None]
                    scale = (
                        (
                            (grouped - x_mean[:, :, None])
                            * q_centered
                            * regression_weight
                        ).sum(dim=2)
                        / (q_centered.square() * regression_weight)
                        .sum(dim=2)
                        .clamp_min_(1e-12)
                    ).clamp_min_(1e-12)
                    minimum = x_mean - scale * q_mean
                    quantized_f32 = torch.round(
                        (grouped - minimum[:, :, None]) / scale[:, :, None]
                    ).clamp_(0, 15)
                if group_importance is None:
                    q_mean = quantized_f32.mean(dim=2)
                    x_mean = grouped.mean(dim=2)
                    regression_weight = 1.0
                else:
                    weight_sum = group_importance.sum(dim=2)
                    q_mean = (quantized_f32 * group_importance).sum(dim=2) / weight_sum
                    x_mean = (grouped * group_importance).sum(dim=2) / weight_sum
                    regression_weight = group_importance
                q_centered = quantized_f32 - q_mean[:, :, None]
                scale = (
                    (
                        (grouped - x_mean[:, :, None])
                        * q_centered
                        * regression_weight
                    ).sum(dim=2)
                    / (q_centered.square() * regression_weight)
                    .sum(dim=2)
                    .clamp_min_(1e-12)
                ).clamp_min_(1e-12)
                minimum = x_mean - scale * q_mean
                if gptq_inverse_cholesky is not None:
                    for gptq_pass in range(2):
                        adjusted = grouped.clone()
                        for column in range(group_size):
                            current = adjusted[:, :, column]
                            code = torch.round(
                                (current - minimum) / scale
                            ).clamp_(0, 15)
                            quantized_f32[:, :, column] = code
                            quantized_value = code * scale + minimum
                            normalized_error = (
                                (current - quantized_value)
                                / gptq_inverse_cholesky[:, column, column][None, :]
                            )
                            adjusted[:, :, column:] -= (
                                normalized_error[:, :, None]
                                * gptq_inverse_cholesky[
                                    :, column, column:
                                ][None, :, :]
                            )
                        if gptq_pass == 0:
                            if group_importance is None:
                                q_mean = quantized_f32.mean(dim=2)
                                x_mean = grouped.mean(dim=2)
                                regression_weight = 1.0
                            else:
                                weight_sum = group_importance.sum(dim=2)
                                q_mean = (
                                    quantized_f32 * group_importance
                                ).sum(dim=2) / weight_sum
                                x_mean = (
                                    grouped * group_importance
                                ).sum(dim=2) / weight_sum
                                regression_weight = group_importance
                            q_centered = quantized_f32 - q_mean[:, :, None]
                            scale = (
                                (
                                    (grouped - x_mean[:, :, None])
                                    * q_centered
                                    * regression_weight
                                ).sum(dim=2)
                                / (q_centered.square() * regression_weight)
                                .sum(dim=2)
                                .clamp_min_(1e-12)
                            ).clamp_min_(1e-12)
                            minimum = x_mean - scale * q_mean
                scales[start:end, :, 0].copy_(scale)
                scales[start:end, :, 1].copy_(minimum)
                quantized = quantized_f32.to(torch.uint8).reshape(end - start, cols)
            else:
                scale = grouped.abs().amax(dim=2).clamp_min_(1e-12).div_(7.0)
                signed = torch.round(grouped / scale[:, :, None]).clamp_(-8, 7)
                # One least-squares refinement materially lowers grouped-INT4
                # error without changing storage or the decode kernel.
                numerator = (grouped * signed).sum(dim=2)
                denominator = signed.square().sum(dim=2).clamp_min_(1.0)
                scale = (numerator / denominator).abs().clamp_min_(1e-12)
                scales[start:end].copy_(scale)
                quantized = (
                    torch.round(grouped / scale[:, :, None])
                    .clamp_(-8, 7)
                    .add_(8)
                    .to(torch.uint8)
                    .reshape(end - start, cols)
                )
        else:
            quantized = torch.empty((end - start, cols), dtype=torch.uint8)
            for group in range(groups):
                col_start = group * group_size
                col_end = min(cols, col_start + group_size)
                values = values_f32[:, col_start:col_end]
                scale = values.abs().amax(dim=1).clamp_min_(1e-12).div_(7.0)
                signed = torch.round(values / scale[:, None]).clamp_(-8, 7)
                numerator = (values * signed).sum(dim=1)
                denominator = signed.square().sum(dim=1).clamp_min_(1.0)
                scale = (numerator / denominator).abs().clamp_min_(1e-12)
                scales[start:end, group].copy_(scale)
                quantized[:, col_start:col_end].copy_(
                    torch.round(values / scale[:, None])
                    .clamp_(-8, 7)
                    .add_(8)
                )
        packed[start:end].copy_(
            quantized[:, 0::2] | (quantized[:, 1::2] << 4)
        )
    return packed, scales


def quantize_mxfp4_to_lowbit_cpu(
    blocks: torch.Tensor,
    scales: torch.Tensor,
    *,
    bits: int,
    chunk_rows: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Requantize packed MXFP4 groups to MSE-selected Q1 or Q2 E8M0 groups."""
    if bits not in {1, 2}:
        raise ValueError("packed MXFP4 expert precision must be Q1 or Q2")
    if blocks.ndim != 4 or scales.ndim != 3:
        raise ValueError("packed MXFP4 blocks/scales must be rank 4/rank 3")
    if tuple(blocks.shape[:3]) != tuple(scales.shape) or blocks.shape[3] != 16:
        raise ValueError(
            f"packed MXFP4 blocks/scales mismatch: {blocks.shape} vs {scales.shape}"
        )
    experts, rows, groups = map(int, scales.shape)
    bytes_per_group = 4 if bits == 1 else 8
    output = torch.empty(
        (experts, rows, groups, bytes_per_group), dtype=torch.uint8
    )
    output_scales = torch.empty_like(scales, dtype=torch.uint8)
    flat_blocks = blocks.reshape(experts * rows, groups, 16)
    flat_scales = scales.reshape(experts * rows, groups)
    flat_output = output.reshape(experts * rows, groups, bytes_per_group)
    flat_output_scales = output_scales.reshape(experts * rows, groups)
    fp4_values = torch.tensor(
        (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
         -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0),
        dtype=torch.float32,
    )
    q2_values = torch.tensor((-1.5, -0.5, 0.5, 1.5), dtype=torch.float32)
    exponent_candidates = (-2, -1, 0, 1, 2, 3)
    for start in range(0, experts * rows, chunk_rows):
        end = min(experts * rows, start + chunk_rows)
        packed = flat_blocks[start:end]
        nibbles = torch.empty((end - start, groups, 32), dtype=torch.uint8)
        nibbles[..., 0::2] = packed & 0x0F
        nibbles[..., 1::2] = packed >> 4
        normalized = fp4_values[nibbles.long()]
        negative = (nibbles & 8) != 0
        best_error: torch.Tensor | None = None
        best_codes: torch.Tensor | None = None
        best_delta: torch.Tensor | None = None
        for delta in exponent_candidates:
            factor = float(2.0**delta)
            if bits == 2:
                outer = normalized.abs() >= factor
                codes = torch.where(
                    negative,
                    torch.where(outer, 0, 1),
                    torch.where(outer, 3, 2),
                ).to(torch.uint8)
                reconstructed = q2_values[codes.long()] * factor
            else:
                codes = (~negative).to(torch.uint8)
                reconstructed = torch.where(negative, -factor, factor)
            error = (normalized - reconstructed).square().mean(dim=2)
            if best_error is None:
                best_error = error
                best_codes = codes
                best_delta = torch.full_like(error, delta, dtype=torch.int16)
            else:
                take = error < best_error
                best_error = torch.where(take, error, best_error)
                best_codes = torch.where(take[..., None], codes, best_codes)
                best_delta = torch.where(
                    take,
                    torch.full_like(best_delta, delta),
                    best_delta,
                )
        assert best_codes is not None and best_delta is not None
        adjusted = (
            flat_scales[start:end].to(torch.int16).sub(127).add(best_delta)
        ).clamp_(-127, 127).add_(127).to(torch.uint8)
        flat_output_scales[start:end].copy_(adjusted)
        if bits == 2:
            flat_output[start:end].copy_(
                best_codes[..., 0::4]
                | (best_codes[..., 1::4] << 2)
                | (best_codes[..., 2::4] << 4)
                | (best_codes[..., 3::4] << 6)
            )
        else:
            packed_signs = best_codes[..., 0::8]
            for bit in range(1, 8):
                packed_signs = packed_signs | (best_codes[..., bit::8] << bit)
            flat_output[start:end].copy_(packed_signs)
    return output, output_scales


def quantize_dense_to_mxfp_lowbit_cpu(
    source: torch.Tensor,
    *,
    bits: int,
    chunk_rows: int = 64,
    compute_device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a dense rank-2 weight to Q1/Q2 groups with E8M0 scales.

    The packed result is ``[rows, cols / 32, bytes_per_group]`` so it can be
    viewed as a one-expert tensor by the existing low-bit tensor-core kernel.
    Each 32-value group chooses the power-of-two scale with minimum squared
    reconstruction error instead of deriving it from a single absolute max.
    """

    if bits not in {1, 2}:
        raise ValueError("dense low-bit precision must be Q1 or Q2")
    if source.ndim != 2:
        raise ValueError("dense low-bit source must be rank 2")
    rows, cols = map(int, source.shape)
    if cols % 32:
        raise ValueError("dense low-bit input width must be divisible by 32")
    if chunk_rows <= 0:
        raise ValueError("dense low-bit chunk_rows must be positive")
    groups = cols // 32
    bytes_per_group = 4 if bits == 1 else 8
    output = torch.empty((rows, groups, bytes_per_group), dtype=torch.uint8)
    output_scales = torch.empty((rows, groups), dtype=torch.uint8)
    exponent_offsets = (-2, -1, 0, 1, 2)
    for start in range(0, rows, chunk_rows):
        end = min(rows, start + chunk_rows)
        values = source[start:end].to(
            device=compute_device or torch.device("cpu"),
            dtype=torch.float32,
        ).reshape(end - start, groups, 32)
        max_abs = values.abs().amax(dim=2).clamp_min_(2.0**-126)
        level_max = 1.5 if bits == 2 else 1.0
        base_exponent = torch.floor(torch.log2(max_abs / level_max)).to(torch.int16)
        best_error: torch.Tensor | None = None
        best_codes: torch.Tensor | None = None
        best_exponent: torch.Tensor | None = None
        for offset in exponent_offsets:
            exponent = (base_exponent + offset).clamp_(-127, 127)
            factor = torch.exp2(exponent.float())
            normalized = values / factor[..., None]
            if bits == 2:
                outer = normalized.abs() >= 1.0
                negative = normalized < 0
                codes = torch.where(
                    negative,
                    torch.where(outer, 0, 1),
                    torch.where(outer, 3, 2),
                ).to(torch.uint8)
                reconstructed = torch.where(
                    outer,
                    torch.copysign(torch.full_like(normalized, 1.5), normalized),
                    torch.copysign(torch.full_like(normalized, 0.5), normalized),
                ) * factor[..., None]
            else:
                codes = (normalized >= 0).to(torch.uint8)
                reconstructed = torch.where(
                    codes.bool(), factor[..., None], -factor[..., None]
                )
            error = (values - reconstructed).square().mean(dim=2)
            if best_error is None:
                best_error = error
                best_codes = codes
                best_exponent = exponent
            else:
                take = error < best_error
                best_error = torch.where(take, error, best_error)
                best_codes = torch.where(take[..., None], codes, best_codes)
                best_exponent = torch.where(take, exponent, best_exponent)
        assert best_codes is not None and best_exponent is not None
        packed_scales = best_exponent.add(127).clamp_(0, 254).to(torch.uint8)
        if bits == 2:
            packed_values = (
                best_codes[..., 0::4]
                | (best_codes[..., 1::4] << 2)
                | (best_codes[..., 2::4] << 4)
                | (best_codes[..., 3::4] << 6)
            )
        else:
            packed_values = torch.zeros(
                (end - start, groups, bytes_per_group),
                dtype=torch.uint8,
                device=values.device,
            )
            for bit_index in range(8):
                packed_values |= best_codes[..., bit_index::8] << bit_index
        output_scales[start:end].copy_(packed_scales.to(device="cpu"))
        output[start:end].copy_(packed_values.to(device="cpu"))
    return output, output_scales


def quantize_mxfp4_to_ternary_cpu(
    blocks: torch.Tensor,
    scales: torch.Tensor,
    *,
    chunk_rows: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack MXFP4 into grouped {-a, 0, +a} weights below two bits/weight.

    Five ternary digits occupy one byte. A three-bit E2M1 magnitude selector is
    bit-packed per 32-weight group, and the source E8M0 group scale is retained.
    The resulting GPT-OSS geometry uses about 1.944 bits/weight including both
    sidecars, without quantizing embeddings, norms, routers, or the LM head.
    """
    if blocks.ndim != 4 or scales.ndim != 3:
        raise ValueError("packed MXFP4 blocks/scales must be rank 4/rank 3")
    if tuple(blocks.shape[:3]) != tuple(scales.shape) or blocks.shape[3] != 16:
        raise ValueError(
            f"packed MXFP4 blocks/scales mismatch: {blocks.shape} vs {scales.shape}"
        )
    experts, rows, groups = map(int, scales.shape)
    cols = groups * 32
    ternary_bytes = (cols + 4) // 5
    magnitude_bytes = (groups * 3 + 7) // 8
    output = torch.zeros(
        (experts, rows, ternary_bytes + magnitude_bytes),
        dtype=torch.uint8,
    )
    flat_blocks = blocks.reshape(experts * rows, groups, 16)
    flat_output = output.reshape(experts * rows, ternary_bytes + magnitude_bytes)
    fp4_values = torch.tensor(
        (
            0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
            -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
        ),
        dtype=torch.float32,
    )
    magnitudes = torch.tensor((0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0))
    powers = (1, 3, 9, 27, 81)
    for start in range(0, experts * rows, chunk_rows):
        end = min(experts * rows, start + chunk_rows)
        packed = flat_blocks[start:end]
        nibbles = torch.empty((end - start, groups, 32), dtype=torch.uint8)
        nibbles[..., 0::2] = packed & 0x0F
        nibbles[..., 1::2] = packed >> 4
        normalized = fp4_values[nibbles.long()]
        best_error: torch.Tensor | None = None
        best_digits: torch.Tensor | None = None
        best_magnitude: torch.Tensor | None = None
        for index, magnitude in enumerate(magnitudes):
            digits = torch.where(
                normalized.abs() > magnitude / 2,
                torch.where(normalized < 0, 0, 2),
                1,
            ).to(torch.uint8)
            reconstructed = torch.where(
                digits == 1,
                0.0,
                torch.where(digits == 0, -magnitude, magnitude),
            )
            error = (normalized - reconstructed).square().mean(dim=2)
            if best_error is None:
                best_error = error
                best_digits = digits
                best_magnitude = torch.full_like(error, index, dtype=torch.uint8)
            else:
                take = error < best_error
                best_error = torch.where(take, error, best_error)
                best_digits = torch.where(take[..., None], digits, best_digits)
                best_magnitude = torch.where(
                    take,
                    torch.full_like(best_magnitude, index),
                    best_magnitude,
                )
        assert best_digits is not None and best_magnitude is not None
        digits = best_digits.reshape(end - start, cols)
        ternary = flat_output[start:end, :ternary_bytes]
        for digit, power in enumerate(powers):
            values = digits[:, digit::5]
            ternary[:, : values.shape[1]].add_(values * power)

        magnitude_output = flat_output[start:end, ternary_bytes:]
        for group in range(groups):
            bit_offset = group * 3
            byte = bit_offset // 8
            shift = bit_offset & 7
            values = best_magnitude[:, group]
            magnitude_output[:, byte] |= values << shift
            if shift > 5:
                magnitude_output[:, byte + 1] |= values >> (8 - shift)
    return output, scales


@dataclass(frozen=True)
class GpuLoadStats:
    archive_open_s: float
    gpu_load_s: float
    pages_loaded: int
    physical_pages_loaded: int
    aliased_pages: int
    fused_logical_pages: int
    unique_gpu_weight_bytes: int
    physical_weight_bytes: int
    cpu_staging_bytes: int
    gpu_transfer_bytes: int
    disk_read_s: float
    cpu_stage_s: float
    gpu_transfer_s: float
    minor_page_faults: int
    major_page_faults: int


@dataclass(frozen=True)
class PageLoadMetrics:
    page_id: str
    bytes: int
    disk_read_s: float
    cpu_stage_s: float
    gpu_transfer_s: float
    cpu_staging_bytes: int
    gpu_transfer_bytes: int
    minor_page_faults: int
    major_page_faults: int


def _ru_faults() -> tuple[int, int]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return int(usage.ru_minflt), int(usage.ru_majflt)


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def _unique_tensor_storage_bytes(tensors: Any) -> int:
    seen: set[tuple[int, int]] = set()
    total = 0
    for tensor in tensors:
        if not isinstance(tensor, torch.Tensor):
            continue
        storage = tensor.untyped_storage()
        key = (storage.data_ptr(), storage.nbytes())
        if key in seen:
            continue
        seen.add(key)
        total += storage.nbytes()
    return total


def _target_dtype(source: torch.dtype, requested: Optional[torch.dtype]) -> torch.dtype:
    return requested if requested is not None and source.is_floating_point else source


def _sync_device(device: torch.device | str) -> None:
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class CpuPageStore:
    def __init__(
        self,
        archive: ThinArchive,
        pin_cpu_pages: bool = False,
        dtype: Optional[torch.dtype] = torch.bfloat16,
    ) -> None:
        self.archive = archive
        self.pin_cpu_pages = pin_cpu_pages
        self.dtype = dtype
        self.tensors: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        self.max_bytes = max(
            1,
            int(
                os.environ.get(
                    "THINTENSOR_CPU_PAGE_CACHE_BYTES",
                    str(2 * 1024**3),
                )
            ),
        )
        self.cpu_pinned_bytes = 0
        self.cpu_resident_bytes = 0
        self.cpu_page_count = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_evictions = 0
        self.cpu_page_source = "bounded_pinned_lru" if pin_cpu_pages else "bounded_lru"

    def get(self, page_id: str) -> torch.Tensor:
        cached = self.tensors.get(page_id)
        if cached is not None:
            self.cache_hits += 1
            self.tensors.move_to_end(page_id)
            return cached
        self.cache_misses += 1
        source, _ = load_tensor_view(self.archive, page_id)
        target_dtype = _target_dtype(source.dtype, self.dtype)
        tensor = torch.empty(
            tuple(source.shape),
            dtype=target_dtype,
            pin_memory=self.pin_cpu_pages,
        )
        tensor.copy_(source, non_blocking=False)
        self.put(page_id, tensor)
        return tensor

    def put(self, page_id: str, tensor: torch.Tensor) -> None:
        previous = self.tensors.pop(page_id, None)
        if previous is not None:
            previous_bytes = _tensor_nbytes(previous)
            self.cpu_resident_bytes -= previous_bytes
            if previous.is_pinned():
                self.cpu_pinned_bytes -= previous_bytes
        incoming = _tensor_nbytes(tensor)
        self._trim(max(0, self.max_bytes - incoming))
        self.tensors[page_id] = tensor
        self.cpu_resident_bytes += incoming
        if tensor.is_pinned():
            self.cpu_pinned_bytes += incoming
        self.cpu_page_count = len(self.tensors)

    def _trim(self, target_bytes: int) -> None:
        while self.tensors and self.cpu_resident_bytes > target_bytes:
            _, tensor = self.tensors.popitem(last=False)
            self.cache_evictions += 1
            size = _tensor_nbytes(tensor)
            self.cpu_resident_bytes -= size
            if tensor.is_pinned():
                self.cpu_pinned_bytes -= size
        self.cpu_page_count = len(self.tensors)


class DirectGpuPageLoader:
    """Loads ThinTensor pages into a GPU allocation through pinned staging."""

    def __init__(
        self,
        archive: ThinArchive,
        device: torch.device,
        dtype: Optional[torch.dtype],
        use_pinned_staging: bool = True,
        cpu_store: Optional[CpuPageStore] = None,
        parent_pool: Optional["ThinGpuPagePool"] = None,
    ) -> None:
        self.archive = archive
        self.device = device
        self.dtype = dtype
        self.use_pinned_staging = use_pinned_staging and device.type == "cuda"
        self.stream = torch.cuda.current_stream(device=device) if device.type == "cuda" else None
        self.prefetch_stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None
        self._pending_staging: deque[tuple[torch.cuda.Event, torch.Tensor]] = deque()
        self.pending_staging_bytes = 0
        self.peak_pending_staging_bytes = 0
        self.resident_bytes_provider: Optional[Callable[[], int]] = None
        self.metrics: list[PageLoadMetrics] = []
        self.cpu_store = cpu_store
        self.parent_pool = parent_pool

    def finish(self) -> None:
        """Wait once for all enqueued page copies, then release pinned staging."""
        if self.stream is not None:
            self.stream.synchronize()
        if self.prefetch_stream is not None:
            self.prefetch_stream.synchronize()
        self._pending_staging.clear()
        self.pending_staging_bytes = 0

    def _reap_completed_staging(self) -> None:
        while self._pending_staging:
            event, staged = self._pending_staging[0]
            try:
                if not event.query():
                    break
            except Exception:
                break
            self._pending_staging.popleft()
            self.pending_staging_bytes -= _tensor_nbytes(staged)

    def load_tensor(self, page: dict[str, Any], stream: Optional[torch.cuda.Stream] = None) -> torch.Tensor:
        page_id = page["id"]
        if self.parent_pool is not None:
            quantized = self.parent_pool._cached_tensor(page_id)
            if quantized is not None:
                tensor, metrics = self._copy_cpu_tensor(
                    page_id,
                    quantized,
                    quantized.dtype,
                    stream=stream,
                )
                self.metrics.append(metrics)
                return tensor
        if self.cpu_store is not None:
            source = self.cpu_store.get(page_id)
            target_dtype = source.dtype
            tensor, metrics = self._copy_cpu_tensor(page_id, source, target_dtype, stream=stream)
            self.metrics.append(metrics)
            return tensor

        source, _ = load_tensor_view(self.archive, page_id)
        target_dtype = _target_dtype(source.dtype, self.dtype)
        tensor, metrics = self._copy_cpu_tensor(page_id, source, target_dtype, stream=stream)
        self.metrics.append(metrics)
        return tensor

    def load_raw_page(self, page_id: str, byte_size: int, stream: Optional[torch.cuda.Stream] = None) -> torch.Tensor:
        if self.cpu_store is not None:
            source = self.cpu_store.get(page_id)
            tensor, metrics = self._copy_cpu_tensor(page_id, source, torch.uint8, stream=stream)
            self.metrics.append(metrics)
            return tensor

        record = self.archive.pages[page_id]
        start_faults = _ru_faults()
        read_start = time.perf_counter()
        if self.archive._mmap is not None:
            with torch.no_grad(), warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="The given buffer is not writable")
                source = torch.frombuffer(
                    self.archive._mmap,
                    dtype=torch.uint8,
                    count=byte_size,
                    offset=record["offset"],
                )
        else:
            source = torch.frombuffer(self.archive.get_page_bytes(page_id), dtype=torch.uint8)
        disk_read_s = time.perf_counter() - read_start
        tensor, metrics = self._copy_cpu_tensor(
            page_id,
            source,
            torch.uint8,
            disk_read_s=disk_read_s,
            start_faults=start_faults,
            stream=stream,
        )
        self.metrics.append(metrics)
        return tensor

    def _copy_cpu_tensor(
        self,
        page_id: str,
        source: torch.Tensor,
        target_dtype: torch.dtype,
        disk_read_s: float = 0.0,
        start_faults: tuple[int, int] | None = None,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> tuple[torch.Tensor, PageLoadMetrics]:
        if start_faults is None:
            start_faults = _ru_faults()

        run_stream = stream if stream is not None else self.stream
        stage_start = time.perf_counter()
        is_source_pinned = source.is_pinned()
        if self.use_pinned_staging and not is_source_pinned:
            staged = torch.empty(
                tuple(source.shape),
                dtype=target_dtype,
                pin_memory=True,
            )
            staged.copy_(source, non_blocking=False)
        elif target_dtype != source.dtype:
            staged = source.to(dtype=target_dtype)
        else:
            staged = source
        cpu_stage_s = time.perf_counter() - stage_start

        transfer_start = time.perf_counter()
        if self.device.type == "cuda" and run_stream is not None:
            self._reap_completed_staging()
            with torch.cuda.stream(run_stream):
                try:
                    gpu = torch.empty_like(staged, device=self.device)
                except torch.OutOfMemoryError as exc:
                    resident_bytes = (
                        self.resident_bytes_provider()
                        if self.resident_bytes_provider is not None
                        else 0
                    )
                    raise RuntimeError(
                        "CUDA OOM loading ThinTensor page "
                        f"{page_id}: shape={tuple(staged.shape)}, "
                        f"dtype={staged.dtype}, "
                        f"attempted_allocation_bytes={_tensor_nbytes(staged)}, "
                        f"resident_bytes={resident_bytes}. "
                        "Use --residency stream and --prefetch 0, or enable "
                        "--lm-head-fp8 when the head is the bottleneck."
                    ) from exc
                gpu.copy_(staged, non_blocking=self.use_pinned_staging or is_source_pinned)
            
            if staged is not source or is_source_pinned:
                copy_done = torch.cuda.Event()
                copy_done.record(run_stream)
                self._pending_staging.append((copy_done, staged))
                self.pending_staging_bytes += _tensor_nbytes(staged)
                self.peak_pending_staging_bytes = max(
                    self.peak_pending_staging_bytes, self.pending_staging_bytes
                )
                if stream is not None and self.parent_pool is not None:
                    self.parent_pool._prefetch_events[page_id] = copy_done
                    self.parent_pool._prefetch_in_progress.add(page_id)
        else:
            gpu = torch.empty_like(staged, device=self.device)
            gpu.copy_(staged, non_blocking=False)
        gpu_transfer_s = time.perf_counter() - transfer_start

        end_faults = _ru_faults()
        metrics = PageLoadMetrics(
            page_id=page_id,
            bytes=_tensor_nbytes(gpu),
            disk_read_s=disk_read_s,
            cpu_stage_s=cpu_stage_s,
            gpu_transfer_s=gpu_transfer_s,
            cpu_staging_bytes=_tensor_nbytes(staged) if staged is not source else 0,
            gpu_transfer_bytes=_tensor_nbytes(gpu),
            minor_page_faults=max(0, end_faults[0] - start_faults[0]),
            major_page_faults=max(0, end_faults[1] - start_faults[1]),
        )
        return gpu, metrics


class ThinGpuWeights:
    """GPU-first ThinTensor weight registry.

    This deliberately does not construct a Hugging Face module and does not use
    load_state_dict. It maps ThinTensor pages, builds CPU views over those pages,
    and immediately materializes the unique tensors into a GPU-owned registry.
    """

    def __init__(
        self,
        archive_path: str | Path,
        device: str = "cuda",
        dtype: Optional[torch.dtype] = None,
        verify: bool = False,
    ) -> None:
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
        self.archive_path = Path(archive_path)
        self.device = torch.device(device)
        self.dtype = dtype
        open_start = time.perf_counter()
        self.archive = ThinArchive(self.archive_path, run_verify=verify)
        self.archive_open_s = time.perf_counter() - open_start
        self.tensors: Dict[str, torch.Tensor] = {}
        self._cached_tensors: Dict[str, torch.Tensor] = {}
        self._cached_optional_tensors: Dict[str, Optional[torch.Tensor]] = {}
        self._cached_rowwise_scales: Dict[str, Optional[torch.Tensor]] = {}
        self._use_tensor_cache = False
        self._rowwise_int8_scales: Dict[str, torch.Tensor] = {}
        self.rowwise_int8_original_bytes = 0
        self.page_specs: Dict[str, dict[str, Any]] = {
            page["id"]: page for page in self.archive.manifest.get("pages", [])
        }
        self.loader = DirectGpuPageLoader(self.archive, self.device, self.dtype)
        self.loader.resident_bytes_provider = lambda: sum(
            _tensor_nbytes(tensor) for tensor in self.tensors.values()
        )
        self.stats = self._load_pages()

    @property
    def manifest(self) -> dict[str, Any]:
        return self.archive.manifest

    @property
    def resident_weight_bytes(self) -> int:
        return _unique_tensor_storage_bytes(
            (
                *self.tensors.values(),
                *self._rowwise_int8_scales.values(),
            )
        )

    def close(self) -> None:
        self.archive.close()

    def _load_pages(self) -> GpuLoadStats:
        load_start = time.perf_counter()
        by_alias: dict[tuple[str, tuple[int, ...], str], torch.Tensor] = {}
        fused_raw: dict[str, torch.Tensor] = {}
        pages_loaded = 0
        physical_pages_loaded = 0
        aliased_pages = 0
        fused_logical_pages = 0
        unique_gpu_weight_bytes = 0
        physical_weight_bytes = 0

        for page in self.manifest.get("pages", []):
            if page.get("kind") != "fused_physical":
                continue
            page_id = page["id"]
            if not self._fused_page_is_referenced(page_id):
                continue
            raw = self.loader.load_raw_page(page_id, int(page["size"]))
            fused_raw[page_id] = raw
            self.tensors[page_id] = raw
            pages_loaded += 1
            physical_pages_loaded += 1
            unique_gpu_weight_bytes += _tensor_nbytes(raw)

        for page in self.manifest.get("pages", []):
            if page.get("kind") == "fused_physical":
                continue
            page_id = page["id"]
            key = (page["checksum"], tuple(page["shape"]), page["dtype"])
            physical_weight_bytes += int(page["size"])
            if key in by_alias:
                self.tensors[page_id] = by_alias[key]
                aliased_pages += 1
                continue

            if page.get("fused_to") is not None and page["fused_to"] in fused_raw:
                gpu_tensor = self._logical_view_from_fused(page, fused_raw[page["fused_to"]])
                fused_logical_pages += 1
            elif str(page.get("op") or "") == "per_layer_token_embeddings":
                gpu_tensor = self._load_rowwise_int8_page(page)
                pages_loaded += 1
                unique_gpu_weight_bytes += _tensor_nbytes(gpu_tensor)
            else:
                gpu_tensor = self.loader.load_tensor(page)
                pages_loaded += 1
                unique_gpu_weight_bytes += _tensor_nbytes(gpu_tensor)
            self.tensors[page_id] = gpu_tensor
            by_alias[key] = gpu_tensor

        self.loader.finish()
        gc.collect()
        malloc_trim()
        loader_metrics = self.loader.metrics
        return GpuLoadStats(
            archive_open_s=self.archive_open_s,
            gpu_load_s=time.perf_counter() - load_start,
            pages_loaded=pages_loaded,
            physical_pages_loaded=physical_pages_loaded,
            aliased_pages=aliased_pages,
            fused_logical_pages=fused_logical_pages,
            unique_gpu_weight_bytes=unique_gpu_weight_bytes,
            physical_weight_bytes=physical_weight_bytes,
            cpu_staging_bytes=sum(metric.cpu_staging_bytes for metric in loader_metrics),
            gpu_transfer_bytes=sum(metric.gpu_transfer_bytes for metric in loader_metrics),
            disk_read_s=sum(metric.disk_read_s for metric in loader_metrics),
            cpu_stage_s=sum(metric.cpu_stage_s for metric in loader_metrics),
            gpu_transfer_s=sum(metric.gpu_transfer_s for metric in loader_metrics),
            minor_page_faults=sum(metric.minor_page_faults for metric in loader_metrics),
            major_page_faults=sum(metric.major_page_faults for metric in loader_metrics),
        )

    def _load_rowwise_int8_page(
        self,
        page: dict[str, Any],
        *,
        chunk_rows: int = 2048,
    ) -> torch.Tensor:
        page_id = str(page["id"])
        source, _ = load_tensor_view(self.archive, page_id)
        if source.ndim != 2:
            raise RuntimeError(
                f"rowwise INT8 page {page_id} must be a matrix"
            )
        rows, cols = (int(source.shape[0]), int(source.shape[1]))
        quantized = torch.empty(
            (rows, cols), device=self.device, dtype=torch.int8
        )
        scales = torch.empty(
            rows, device=self.device, dtype=torch.float32
        )
        self.rowwise_int8_original_bytes += _tensor_nbytes(source)
        for start in range(0, rows, chunk_rows):
            end = min(rows, start + chunk_rows)
            chunk = source[start:end].to(
                device=self.device,
                dtype=torch.bfloat16,
            )
            scale = (
                chunk.abs()
                .amax(dim=1)
                .float()
                .clamp_min_(1e-12)
                .div_(127.0)
            )
            scales[start:end].copy_(scale)
            quantized[start:end].copy_(
                torch.round(chunk.float() / scale[:, None])
                .clamp_(-127, 127)
                .to(torch.int8)
            )
            del chunk, scale
        self._rowwise_int8_scales[page_id] = scales
        return quantized

    def rowwise_int8_scale(self, page_id: str) -> Optional[torch.Tensor]:
        if getattr(self, "_use_tensor_cache", False):
            try:
                return self._cached_rowwise_scales[page_id]
            except KeyError:
                pass
        val = self._rowwise_int8_scales.get(page_id)
        if getattr(self, "_use_tensor_cache", False):
            self._cached_rowwise_scales[page_id] = val
        return val

    def tensor(self, page_id: str) -> torch.Tensor:
        if getattr(self, "_use_tensor_cache", False):
            try:
                return self._cached_tensors[page_id]
            except KeyError:
                pass
        try:
            val = self.tensors[page_id]
            if getattr(self, "_use_tensor_cache", False):
                self._cached_tensors[page_id] = val
            return val
        except KeyError as exc:
            raise KeyError(f"ThinTensor GPU page {page_id} is not loaded") from exc

    def _fused_page_is_referenced(self, fused_id: str) -> bool:
        return any(page.get("fused_to") == fused_id for page in self.manifest.get("pages", []))

    def _logical_view_from_fused(self, page: dict[str, Any], raw: torch.Tensor) -> torch.Tensor:
        source_dtype = map_dtype(page["dtype"])
        offset = int(page.get("fused_offset", 0))
        size = int(page["size"])
        if offset < 0 or offset + size > raw.numel():
            raise RuntimeError(f"fused slice for {page['id']} is outside {page['fused_to']}")
        byte_slice = raw.narrow(0, offset, size)
        try:
            typed = byte_slice.view(source_dtype)
        except RuntimeError as exc:
            raise RuntimeError(
                f"cannot view fused GPU page {page['fused_to']} slice for {page['id']} as {source_dtype}"
            ) from exc
        tensor = typed.reshape(tuple(int(dim) for dim in page["shape"]))
        target_dtype = _target_dtype(source_dtype, self.dtype)
        if target_dtype != source_dtype:
            tensor = tensor.to(dtype=target_dtype)
        return tensor


class ThinGpuPagePool:
    """Execution-tape aware VRAM page pool for streaming residency."""

    def __init__(
        self,
        archive_path: str | Path,
        device: str = "cuda",
        dtype: Optional[torch.dtype] = torch.bfloat16,
        vram_budget_bytes: Optional[int] = None,
        prefetch_distance: int = 1,
        verify: bool = False,
        cpu_offload: bool = False,
        pin_cpu_pages: bool = False,
        debug_stream_refs: bool = False,
        down_proj_fp8: bool = False,
        gate_up_fp8: bool = False,
        qkv_fp8: bool = False,
        o_proj_fp8: bool = False,
        embed_fp8: bool = False,
        cpu_embed: bool = False,
        fp8_layer_spec: Optional[str] = None,
        down_fp8_layer_spec: Optional[str] = None,
        qkv_fp8_layer_spec: Optional[str] = None,
        o_fp8_layer_spec: Optional[str] = None,
        fp8_scale_block: int = 0,
        lm_head_fp8: bool = False,
        lm_head_fp8_scale_block: int = 0,
        lm_head_int4_group_size: int = 0,
        expert_int4: bool = False,
        expert_int4_layer_spec: Optional[str] = None,
        expert_int4_group_size: int = 32,
        dense_int4: bool = False,
        dense_int4_layer_spec: Optional[str] = None,
        dense_int4_group_size: int = 32,
        dense_int4_attention_group_size: int = 0,
        dense_int4_fine_layer_spec: Optional[str] = None,
        dense_int4_fine_group_size: int = 0,
        dense_int4_cpu_layer_spec: Optional[str] = None,
        dense_int4_calibration_json: Optional[str] = None,
        dense_int4_calibration_npz: Optional[str] = None,
        dense_int4_calibration_covariance_npz: Optional[str] = None,
        dense_lowbit_bits: int = 0,
        dense_lowbit_layer_spec: Optional[str] = None,
        mxfp4_gate_up_layer_spec: Optional[str] = None,
        mxfp4_down_layer_spec: Optional[str] = None,
        mxfp4_qkv_layer_spec: Optional[str] = None,
        mxfp4_o_layer_spec: Optional[str] = None,
        packed_expert_q2_layer_spec: Optional[str] = None,
        packed_expert_q1_layer_spec: Optional[str] = None,
    ) -> None:
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
        self.archive_path = Path(archive_path)
        self.device = torch.device(device)
        self.dtype = dtype
        self.vram_budget_bytes = vram_budget_bytes
        self.prefetch_distance = max(0, prefetch_distance)
        self.cpu_offload = cpu_offload
        self.pin_cpu_pages = pin_cpu_pages
        self.debug_stream_refs = debug_stream_refs
        self.down_proj_fp8 = down_proj_fp8
        self.gate_up_fp8 = gate_up_fp8
        self.qkv_fp8 = qkv_fp8
        self.o_proj_fp8 = o_proj_fp8
        self.embed_fp8 = embed_fp8
        self.cpu_embed = bool(cpu_embed)
        if self.cpu_embed and self.embed_fp8:
            raise ValueError("cpu_embed and embed_fp8 are mutually exclusive")
        self.fp8_layer_spec = fp8_layer_spec
        self.fp8_scale_block = fp8_scale_block
        self.lm_head_fp8 = bool(lm_head_fp8)
        self.lm_head_fp8_scale_block = lm_head_fp8_scale_block
        self.lm_head_int4_group_size = int(lm_head_int4_group_size)
        self.expert_int4 = bool(expert_int4)
        self.expert_int4_group_size = int(expert_int4_group_size)
        self.dense_int4 = bool(dense_int4)
        self.dense_int4_group_size = int(dense_int4_group_size)
        self.dense_int4_attention_group_size = int(
            dense_int4_attention_group_size
        )
        self.dense_int4_fine_group_size = int(dense_int4_fine_group_size)
        if bool(dense_int4_calibration_json) != bool(dense_int4_calibration_npz):
            raise ValueError(
                "dense INT4 calibration requires both the JSON mapping and NPZ data"
            )
        self._dense_int4_input_rms: Dict[str, torch.Tensor] = {}
        self._dense_int4_covariance_path = dense_int4_calibration_covariance_npz
        self._dense_int4_covariance_names: Dict[str, str] = {}
        if dense_int4_calibration_json:
            calibration_meta = json.loads(
                Path(dense_int4_calibration_json).read_text(encoding="utf-8")
            )
            mapping = calibration_meta.get("input_stats_mapping", {})
            with np.load(dense_int4_calibration_npz) as calibration_arrays:
                for page_id, names in mapping.items():
                    rms_name = names.get("rms")
                    if rms_name in calibration_arrays:
                        self._dense_int4_input_rms[page_id] = torch.from_numpy(
                            calibration_arrays[rms_name].copy()
                        )
            if dense_int4_calibration_covariance_npz:
                with np.load(dense_int4_calibration_covariance_npz) as arrays:
                    available = set(arrays.files)
                for page_id, names in mapping.items():
                    covariance_name = names.get("covariance32")
                    if covariance_name in available:
                        self._dense_int4_covariance_names[page_id] = covariance_name
        self.dense_lowbit_bits = int(dense_lowbit_bits)
        if self.dense_lowbit_bits not in {0, 1, 2}:
            raise ValueError("dense_lowbit_bits must be 0, 1, or 2")
        self.decode_steps_run = 0

        open_start = time.perf_counter()
        self.archive = ThinArchive(self.archive_path, run_verify=verify)
        self.archive_open_s = time.perf_counter() - open_start
        self.manifest = self.archive.manifest
        selected_fp8_layers = _parse_layer_selection(
            fp8_layer_spec,
            int(self.manifest["model"]["layers"]),
        )
        self.selected_fp8_layers = selected_fp8_layers
        selected_down_fp8_layers = _parse_layer_selection(
            down_fp8_layer_spec
            if down_fp8_layer_spec is not None
            else fp8_layer_spec,
            int(self.manifest["model"]["layers"]),
        )
        self.selected_down_fp8_layers = selected_down_fp8_layers
        selected_qkv_fp8_layers = _parse_layer_selection(
            qkv_fp8_layer_spec
            if qkv_fp8_layer_spec is not None
            else fp8_layer_spec,
            int(self.manifest["model"]["layers"]),
        )
        self.selected_qkv_fp8_layers = selected_qkv_fp8_layers
        selected_o_fp8_layers = _parse_layer_selection(
            o_fp8_layer_spec
            if o_fp8_layer_spec is not None
            else fp8_layer_spec,
            int(self.manifest["model"]["layers"]),
        )
        self.selected_o_fp8_layers = selected_o_fp8_layers
        selected_expert_int4_layers = _parse_layer_selection(
            expert_int4_layer_spec,
            int(self.manifest["model"]["layers"]),
        )
        self.selected_expert_int4_layers = selected_expert_int4_layers
        selected_dense_int4_layers = (
            _parse_layer_selection(
                dense_int4_layer_spec,
                int(self.manifest["model"]["layers"]),
            )
            if dense_int4
            else set()
        )
        self.selected_dense_int4_cpu_layers = _parse_layer_selection(
            dense_int4_cpu_layer_spec,
            int(self.manifest["model"]["layers"]),
        )
        if self.selected_dense_int4_cpu_layers:
            self.dense_int4 = True
            selected_dense_int4_layers |= self.selected_dense_int4_cpu_layers
        self._dense_int4_gpu_ops = {
            item.strip()
            for item in os.environ.get("THINTENSOR_DENSE_INT4_GPU_OPS", "").split(",")
            if item.strip()
        }
        self._dense_int4_gpu_suffixes = tuple(
            item.strip()
            for item in os.environ.get(
                "THINTENSOR_DENSE_INT4_GPU_SUFFIXES", ""
            ).split(",")
            if item.strip()
        )
        self.selected_dense_int4_layers = selected_dense_int4_layers
        selected_dense_int4_fine_layers = _parse_layer_selection(
            dense_int4_fine_layer_spec,
            int(self.manifest["model"]["layers"]),
        ) if dense_int4_fine_layer_spec is not None else set()
        self.selected_dense_lowbit_layers = _parse_layer_selection(
            dense_lowbit_layer_spec,
            int(self.manifest["model"]["layers"]),
        )
        self._dense_lowbit_suffixes = tuple(
            item.strip()
            for item in os.environ.get(
                "THINTENSOR_DENSE_LOWBIT_SUFFIXES", ""
            ).split(",")
            if item.strip()
        )
        model_layers = int(self.manifest["model"]["layers"])
        self.selected_mxfp4_gate_up_layers = (
            _parse_layer_selection(mxfp4_gate_up_layer_spec, model_layers)
            if mxfp4_gate_up_layer_spec is not None else set()
        )
        self.selected_mxfp4_down_layers = (
            _parse_layer_selection(mxfp4_down_layer_spec, model_layers)
            if mxfp4_down_layer_spec is not None else set()
        )
        self.selected_mxfp4_qkv_layers = (
            _parse_layer_selection(mxfp4_qkv_layer_spec, model_layers)
            if mxfp4_qkv_layer_spec is not None else set()
        )
        self.selected_mxfp4_o_layers = (
            _parse_layer_selection(mxfp4_o_layer_spec, model_layers)
            if mxfp4_o_layer_spec is not None else set()
        )
        self.selected_packed_expert_q2_layers = _parse_layer_selection(
            packed_expert_q2_layer_spec,
            int(self.manifest["model"]["layers"]),
        )
        self.selected_packed_expert_q1_layers = _parse_layer_selection(
            packed_expert_q1_layer_spec,
            int(self.manifest["model"]["layers"]),
        )
        self.page_specs: Dict[str, dict[str, Any]] = {
            page["id"]: {
                **page,
                "shape": list(page.get("shape", ())),
            }
            for page in self.manifest.get("pages", [])
        }

        self._fp8_cpu_cache: Dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._int4_cpu_cache: Dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._cpu_int4_pages: set[str] = set()
        self._cpu_int4_kernel_cache: Dict[
            str, tuple[torch.Tensor, torch.Tensor]
        ] = {}
        self._cpu_int4_residual_terms = max(
            0, int(os.environ.get("THINTENSOR_CPU_INT4_RESIDUAL_TERMS", "0"))
        )
        self._cpu_int4_residual_sidecars: Dict[
            str, tuple[torch.Tensor, torch.Tensor]
        ] = {}
        self._int4_source_pages: set[str] = set()
        self._mxfp4_source_pages: set[str] = set()
        self._mxfp4_cpu_cache: Dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._mxfp4_metadata_by_page: Dict[str, tuple[int, int, int]] = {}
        self._expert_int4_pack_cpu: Dict[str, torch.Tensor] = {}
        self._expert_int4_pack_page_ids: Dict[
            int, tuple[str, str, str, str]
        ] = {}
        self._int4_metadata_by_page: Dict[str, tuple[int, int, int]] = {}
        self._int4_cache_paths: Dict[str, Path] = {}
        self._dense_lowbit_source_pages: set[str] = set()
        self._dense_lowbit_cpu_cache: Dict[
            str, tuple[torch.Tensor, torch.Tensor]
        ] = {}
        self._dense_lowbit_metadata_by_page: Dict[
            str, tuple[int, int, int]
        ] = {}
        self._dense_lowbit_cache_paths: Dict[str, Path] = {}
        self._weight_scales: Dict[str, torch.Tensor] = {}
        self._packed_lowbit_by_page: Dict[str, tuple[str, str, int]] = {}
        self._packed_lowbit_cache: Dict[
            str, tuple[torch.Tensor, torch.Tensor]
        ] = {}
        self._packed_lowbit_cache_paths: Dict[str, Path] = {}
        self.derived_cache_hits = 0
        self.derived_cache_misses = 0
        self.derived_cache_writes = 0
        self.derived_cache_bytes = 0
        self._derived_cache_enabled = (
            os.environ.get("THINTENSOR_DERIVED_CACHE", "1") != "0"
        )
        self._derived_cache_dir = self.archive_path.parent / ".derived"
        self._quantization_lock = threading.Lock()
        self._tensor_id_to_page_id: Dict[int, str] = {}
        self._compact_expert_index_cache: Dict[int, torch.Tensor] = {}
        self._cpu_embed_source: Optional[torch.Tensor] = None
        self._cpu_embed_meta: Optional[torch.Tensor] = None

        # The execution head is a global page, so residency planning must see
        # its packed size before choosing resident layers.  Quantizing it only
        # after warm_start leaves a BF16-sized hole in the plan and causes a
        # nearly-fit model to stream several body layers forever.
        if self.lm_head_int4_group_size > 0 and "lm_head.weight" in self.page_specs:
            page = self.page_specs["lm_head.weight"]
            shape = page.get("shape") or ()
            if len(shape) != 2:
                raise ValueError("INT4 lm_head requires a rank-2 weight")
            rows, cols = int(shape[0]), int(shape[1])
            if cols % 2:
                raise ValueError("INT4 lm_head requires an even input width")
            groups = (cols + self.lm_head_int4_group_size - 1) // self.lm_head_int4_group_size
            self._int4_source_pages.add("lm_head.weight")
            self._int4_metadata_by_page["lm_head.weight"] = (
                rows,
                cols,
                self.lm_head_int4_group_size,
            )
            page["shape"] = [rows, cols // 2]
            page["dtype"] = "torch.uint8"
            page["size"] = rows * (cols // 2)
            page["quant_scheme"] = "runtime_grouped_int4"
            page["bits_per_weight"] = 4
            page["quant_group_size"] = self.lm_head_int4_group_size
            scale_id = "lm_head.weight.scale"
            page["scale_page"] = scale_id
            self.page_specs[scale_id] = {
                "id": scale_id,
                "shape": [rows, groups],
                "dtype": "torch.float16",
                "size": rows * groups * 2,
                "kind": "scale",
                "layer": None,
                "checksum": "runtime-int4-scale:lm_head.weight",
            }

        for page_id, page in list(self.page_specs.items()):
            layer = page.get("layer")
            if layer is None or not is_expert_weight(page):
                continue
            layer = int(layer)
            bits = (
                1 if layer in self.selected_packed_expert_q1_layers
                else 2 if layer in self.selected_packed_expert_q2_layers
                else 0
            )
            if bits == 0 or int(page.get("bits_per_weight") or 0) != 4:
                continue
            scale_id = str(page.get("scale_page") or "")
            if not scale_id or scale_id not in self.page_specs:
                raise RuntimeError(
                    f"packed expert page {page_id} has no E8M0 scale page"
                )
            if len(page["shape"]) != 4 or int(page["shape"][-1]) != 16:
                raise RuntimeError(
                    f"packed expert page {page_id} has unsupported shape {page['shape']}"
                )
            source_checksum = str(page.get("checksum") or "")
            scale_checksum = str(
                self.page_specs[scale_id].get("checksum") or ""
            )
            cache_key = hashlib.sha256(
                (
                    "runtime-mxfp-lowbit-v2\0"
                    + source_checksum
                    + "\0"
                    + scale_checksum
                    + f"\0q{bits}"
                ).encode("utf-8")
            ).hexdigest()
            self._packed_lowbit_cache_paths[page_id] = (
                self._derived_cache_dir / f"{cache_key}.pt"
            )
            page["shape"][-1] = 4 if bits == 1 else 8
            page["size"] = math.prod(page["shape"])
            page["bits_per_weight"] = bits
            page["quant_scheme"] = f"runtime_mxfp_q{bits}"
            page["checksum"] = f"runtime-mxfp-q{bits}:{page_id}"
            binding = (page_id, scale_id, bits)
            self._packed_lowbit_by_page[page_id] = binding
            self._packed_lowbit_by_page[scale_id] = binding

        if (
            self.down_proj_fp8
            or self.gate_up_fp8
            or self.qkv_fp8
            or self.o_proj_fp8
            or self.embed_fp8
            or self.lm_head_fp8
        ):
            import sys
            print("Pre-quantizing selected weights to scaled FP8...", file=sys.stderr)
            for page_id, page in list(self.page_specs.items()):
                page_layer = page.get("layer")
                layer_selected = (
                    page_layer is not None
                    and int(page_layer) in selected_fp8_layers
                )
                role = page_role(page)
                op = str(page.get("op") or "")
                is_down = (
                    op == "mlp_down_proj"
                    and self.down_proj_fp8
                    and page_layer is not None
                    and int(page_layer) in selected_down_fp8_layers
                )
                is_gate_up = (
                    op in {"mlp_gate_proj", "mlp_up_proj", "mlp_gate_up_proj"}
                    and self.gate_up_fp8
                )
                is_qkv = (
                    op in {
                        "attn_q_proj", "attn_k_proj", "attn_v_proj",
                        "attn_qkv_proj", "linear_attn_input_projection",
                    }
                    and self.qkv_fp8
                    and page_layer is not None
                    and int(page_layer) in selected_qkv_fp8_layers
                )
                is_o = (
                    op in {"attn_o_proj", "linear_attn_output_projection"}
                    and self.o_proj_fp8
                    and page_layer is not None
                    and int(page_layer) in selected_o_fp8_layers
                )
                is_embed = (
                    self.embed_fp8
                    and page_id == "model.embed_tokens.weight"
                )
                is_lm_head = self.lm_head_fp8 and page_id == "lm_head.weight"
                if is_embed or is_lm_head or is_down or is_qkv or is_o or (
                    layer_selected and is_gate_up
                ):
                    source, _ = load_tensor_view(self.archive, page_id)
                    q_w, q_s = quantize_to_fp8_via_gpu(
                        source,
                        device=self.device,
                        scale_block_size=self.fp8_scale_block,
                    )
                    
                    self._fp8_cpu_cache[page_id] = (q_w, q_s)
                    
                    page["dtype"] = "torch.float8_e4m3fn"
                    page["size"] = q_w.numel()
                    page["quant_scheme"] = "runtime_row_fp8"
                    
                    scale_id = page_id + ".scale"
                    page["scale_page"] = scale_id
                    self.page_specs[scale_id] = {
                        "id": scale_id,
                        "shape": list(q_s.shape),
                        "dtype": "torch.float32",
                        "size": q_s.numel() * 4,
                        "kind": "scale",
                        "layer": page.get("layer"),
                        "checksum": f"runtime-scale:{page_id}",
                    }

        if self.expert_int4:
            import sys
            print(
                "Configuring router-driven lazy INT4 for middle-layer experts...",
                file=sys.stderr,
            )
            for page_id, page in list(self.page_specs.items()):
                page_layer = page.get("layer")
                if (
                    page_layer is None
                    or int(page_layer) not in selected_expert_int4_layers
                    or not is_expert_weight(page)
                ):
                    continue
                rows, cols = int(page["shape"][0]), int(page["shape"][1])
                groups = (
                    cols + self.expert_int4_group_size - 1
                ) // self.expert_int4_group_size
                self._int4_source_pages.add(page_id)
                self._int4_metadata_by_page[page_id] = (
                    rows,
                    cols,
                    self.expert_int4_group_size,
                )
                page["shape"] = [rows, cols // 2]
                page["dtype"] = "torch.uint8"
                page["size"] = rows * (cols // 2)
                page["quant_scheme"] = "runtime_grouped_int4"
                page["bits_per_weight"] = 4
                page["quant_group_size"] = self.expert_int4_group_size
                scale_id = page_id + ".scale"
                page["scale_page"] = scale_id
                affine_int4 = os.environ.get("THINTENSOR_AFFINE_INT4", "0") == "1"
                self.page_specs[scale_id] = {
                    "id": scale_id,
                    "shape": [rows, groups, 2] if affine_int4 else [rows, groups],
                    "dtype": "torch.float16",
                    "size": rows * groups * (4 if affine_int4 else 2),
                    "kind": "scale",
                    "layer": page_layer,
                    "checksum": f"runtime-int4-scale:{page_id}",
                }
            num_experts = int(
                self.manifest["model"].get("num_local_experts") or 0
            )
            for layer in sorted(selected_expert_int4_layers):
                if num_experts <= 0:
                    break
                gate_id, up_id, down_id = _separate_expert_tensor_ids(
                    self, layer, 0
                )
                gate_rows, gate_cols, _ = self._int4_metadata_by_page[gate_id]
                up_rows, up_cols, _ = self._int4_metadata_by_page[up_id]
                down_rows, down_cols, _ = self._int4_metadata_by_page[down_id]
                if gate_rows != up_rows or gate_cols != up_cols:
                    raise RuntimeError(
                        f"layer {layer} expert gate/up geometry differs"
                    )
                prefix = f"__runtime__.layer.{layer}.experts"
                ids = (
                    prefix + ".gate_up.int4",
                    prefix + ".gate_up.int4.scale",
                    prefix + ".down.int4",
                    prefix + ".down.int4.scale",
                )
                self._expert_int4_pack_page_ids[layer] = ids
                groups_gate = (
                    gate_cols + self.expert_int4_group_size - 1
                ) // self.expert_int4_group_size
                groups_down = (
                    down_cols + self.expert_int4_group_size - 1
                ) // self.expert_int4_group_size
                specs = (
                    (ids[0], [num_experts, gate_rows + up_rows, gate_cols // 2], "torch.uint8"),
                    (ids[1], [num_experts, gate_rows + up_rows, groups_gate], "torch.float32"),
                    (ids[2], [num_experts, down_rows, down_cols // 2], "torch.uint8"),
                    (ids[3], [num_experts, down_rows, groups_down], "torch.float32"),
                )
                for pack_id, shape, dtype_name in specs:
                    element_size = 4 if dtype_name == "torch.float32" else 1
                    self.page_specs[pack_id] = {
                        "id": pack_id,
                        "shape": shape,
                        "dtype": dtype_name,
                        "size": math.prod(shape) * element_size,
                        "kind": "expert_pack",
                        "layer": layer,
                        "checksum": f"runtime-expert-pack:{pack_id}",
                    }

        if self.dense_lowbit_bits:
            import sys

            print(
                f"Configuring lazy dense Q{self.dense_lowbit_bits} E8M0 projections...",
                file=sys.stderr,
            )
            bytes_per_group = 4 if self.dense_lowbit_bits == 1 else 8
            for page_id, page in list(self.page_specs.items()):
                page_layer = page.get("layer")
                shape = page.get("shape") or ()
                if (
                    page_layer is None
                    or (
                        int(page_layer) not in self.selected_dense_lowbit_layers
                        and not any(
                            page_id.endswith(suffix)
                            for suffix in self._dense_lowbit_suffixes
                        )
                    )
                    or len(shape) != 2
                    or not (is_body_weight(page) and not is_expert_weight(page))
                    or page_id in self._fp8_cpu_cache
                ):
                    continue
                rows, cols = int(shape[0]), int(shape[1])
                if cols % 32:
                    continue
                groups = cols // 32
                self._dense_lowbit_source_pages.add(page_id)
                self._dense_lowbit_metadata_by_page[page_id] = (
                    rows,
                    cols,
                    self.dense_lowbit_bits,
                )
                cache_key = hashlib.sha256(
                    (
                        "runtime-dense-mxfp-lowbit-v2\0"
                        + str(page.get("checksum") or "")
                        + f"\0q{self.dense_lowbit_bits}"
                    ).encode("utf-8")
                ).hexdigest()
                self._dense_lowbit_cache_paths[page_id] = (
                    self._derived_cache_dir / f"{cache_key}.pt"
                )
                page["shape"] = [rows, groups, bytes_per_group]
                page["dtype"] = "torch.uint8"
                page["size"] = rows * groups * bytes_per_group
                page["quant_scheme"] = (
                    f"runtime_dense_mxfp_q{self.dense_lowbit_bits}_e8m0"
                )
                page["bits_per_weight"] = self.dense_lowbit_bits
                page["quant_group_size"] = 32
                scale_id = page_id + ".scale"
                page["scale_page"] = scale_id
                self.page_specs[scale_id] = {
                    "id": scale_id,
                    "shape": [rows, groups],
                    "dtype": "torch.uint8",
                    "size": rows * groups,
                    "kind": "scale",
                    "layer": page_layer,
                    "checksum": f"runtime-dense-q{self.dense_lowbit_bits}-scale:{page_id}",
                }

        selected_mxfp4_pages: set[str] = set()
        if any((
            self.selected_mxfp4_gate_up_layers,
            self.selected_mxfp4_down_layers,
            self.selected_mxfp4_qkv_layers,
            self.selected_mxfp4_o_layers,
        )):
            import sys
            print(
                "Configuring lazy native MXFP4 for selected dense projections...",
                file=sys.stderr,
            )
            for page_id, page in list(self.page_specs.items()):
                page_layer = page.get("layer")
                shape = page.get("shape") or ()
                if page_layer is None or len(shape) != 2:
                    continue
                layer = int(page_layer)
                gate_up = any(
                    page_id.endswith(suffix)
                    for suffix in (
                        "mlp.gate_proj.weight",
                        "mlp.up_proj.weight",
                        "mlp.gate_up_proj.weight",
                    )
                ) and layer in self.selected_mxfp4_gate_up_layers
                down = page_id.endswith("mlp.down_proj.weight") and (
                    layer in self.selected_mxfp4_down_layers
                )
                qkv = (
                    any(
                        token in page_id
                        for token in (
                            "self_attn.q_proj.weight",
                            "self_attn.k_proj.weight",
                            "self_attn.v_proj.weight",
                            "self_attn.qkv_proj.weight",
                            "linear_attn.in_proj_",
                        )
                    )
                    and layer in self.selected_mxfp4_qkv_layers
                )
                out = (
                    page_id.endswith("self_attn.o_proj.weight")
                    or page_id.endswith("linear_attn.out_proj.weight")
                ) and layer in self.selected_mxfp4_o_layers
                if not (gate_up or down or qkv or out):
                    continue
                rows, cols = int(shape[0]), int(shape[1])
                if cols % 32:
                    continue
                selected_mxfp4_pages.add(page_id)
                self._mxfp4_source_pages.add(page_id)
                self._mxfp4_metadata_by_page[page_id] = (rows, cols, 0)
                page["shape"] = [rows, cols // 2]
                page["dtype"] = "torch.uint8"
                page["size"] = rows * (cols // 2)
                page["quant_scheme"] = "runtime_mxfp4_e8m0"
                page["bits_per_weight"] = 4
                scale_id = page_id + ".scale"
                page["scale_page"] = scale_id
                self.page_specs[scale_id] = {
                    "id": scale_id,
                    "shape": [rows, cols // 32],
                    "dtype": "torch.uint8",
                    "size": rows * (cols // 32),
                    "kind": "scale",
                    "layer": page_layer,
                    "checksum": f"runtime-mxfp4-scale:{page_id}",
                }

        if self.dense_int4:
            import sys
            print(
                "Configuring router-driven lazy INT4 for middle-layer dense projections...",
                file=sys.stderr,
            )
            for page_id, page in list(self.page_specs.items()):
                page_layer = page.get("layer")
                if (
                    page_layer is None
                    or int(page_layer) not in self.selected_dense_int4_layers
                    or not (is_body_weight(page) and not is_expert_weight(page))
                    or page_id in selected_mxfp4_pages
                    or page_id in self._dense_lowbit_source_pages
                    or page_id in self._fp8_cpu_cache
                ):
                    continue
                rows, cols = int(page["shape"][0]), int(page["shape"][1])
                attention_ops = {
                    "attn_q_proj", "attn_k_proj", "attn_v_proj",
                    "attn_qkv_proj", "attn_o_proj",
                    "linear_attn_input_projection",
                    "linear_attn_output_projection",
                }
                group_size = self.dense_int4_group_size
                if (
                    self.dense_int4_fine_group_size > 0
                    and int(page_layer) in selected_dense_int4_fine_layers
                    and str(page.get("op") or "").startswith("mlp_")
                ):
                    group_size = self.dense_int4_fine_group_size
                if (
                    self.dense_int4_attention_group_size > 0
                    and str(page.get("op") or "") in attention_ops
                ):
                    group_size = self.dense_int4_attention_group_size
                groups = (cols + group_size - 1) // group_size
                self._int4_source_pages.add(page_id)
                if (
                    int(page_layer) in self.selected_dense_int4_cpu_layers
                    and str(page.get("op") or "") not in self._dense_int4_gpu_ops
                    and not any(
                        page_id.endswith(suffix)
                        for suffix in self._dense_int4_gpu_suffixes
                    )
                ):
                    self._cpu_int4_pages.add(page_id)
                self._int4_metadata_by_page[page_id] = (
                    rows,
                    cols,
                    group_size,
                )
                calibration_identity = "uncalibrated"
                if dense_int4_calibration_json:
                    calibration_identity = "|".join(
                        (
                            str(calibration_meta.get("input_stats_sha256") or ""),
                            str(calibration_meta.get("input_covariance_sha256") or ""),
                            os.environ.get("THINTENSOR_GPTQ_DAMP_PERCENT", "0.01"),
                        )
                    )
                cache_key = hashlib.sha256(
                    (
                        "runtime-dense-int4-v4\0"
                        + str(page.get("checksum") or "")
                        + f"\0g{group_size}\0affine={int(os.environ.get('THINTENSOR_AFFINE_INT4', '0') == '1')}"
                        + "\0" + calibration_identity
                    ).encode("utf-8")
                ).hexdigest()
                self._int4_cache_paths[page_id] = (
                    self._derived_cache_dir / f"{cache_key}.pt"
                )
                page["shape"] = [rows, cols // 2]
                page["dtype"] = "torch.uint8"
                page["size"] = rows * (cols // 2)
                page["quant_scheme"] = "runtime_grouped_int4"
                page["bits_per_weight"] = 4
                page["quant_group_size"] = group_size
                scale_id = page_id + ".scale"
                page["scale_page"] = scale_id
                affine_page = (
                    os.environ.get("THINTENSOR_AFFINE_INT4", "0") == "1"
                    and rows < 65536
                )
                self.page_specs[scale_id] = {
                    "id": scale_id,
                    "shape": [rows, groups, 2] if affine_page else [rows, groups],
                    "dtype": "torch.float16",
                    "size": rows * groups * (4 if affine_page else 2),
                    "kind": "scale",
                    "layer": page_layer,
                    "checksum": f"runtime-int4-scale:{page_id}",
                }

        self.cpu_store = None
        if self.cpu_offload:
            self.cpu_store = CpuPageStore(
                self.archive, pin_cpu_pages=self.pin_cpu_pages, dtype=self.dtype
            )
            for page_id, (q_w, q_s) in self._fp8_cpu_cache.items():
                self.cpu_store.put(page_id, q_w)
                self.cpu_store.put(page_id + ".scale", q_s)
            for page_id, (q_w, q_s) in self._int4_cpu_cache.items():
                self.cpu_store.put(page_id, q_w)
                self.cpu_store.put(page_id + ".scale", q_s)

        self.loader = DirectGpuPageLoader(
            self.archive,
            self.device,
            self.dtype,
            use_pinned_staging=(
                os.environ.get("THINTENSOR_PINNED_STAGING", "1") != "0"
            ),
            cpu_store=self.cpu_store,
            parent_pool=self,
        )
        self.tensors: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        self.owned_bytes: dict[str, int] = {}
        self._active_pages: set[str] = set()
        self._page_use_events: dict[str, torch.cuda.Event] = {}
        self._retired_tensors: deque[
            tuple[torch.cuda.Event, torch.Tensor, int]
        ] = deque()
        self._retired_tensor_bytes = 0
        self.alias_keys: dict[tuple[str, tuple[int, ...], str], str] = {}
        self.resident_bytes = 0
        self.external_resident_bytes = 0
        self.loader.resident_bytes_provider = lambda: self.resident_bytes
        
        self.streamed_packed_expert_pages: set[str] = set()
        self.streamed_packed_expert_bytes = 0
        self.promoted_packed_expert_layers: list[int] = []
        self.packed_expert_promotion_ooms = 0
        self.mxfp4_expert_slice_cache_limit_bytes = int(
            os.environ.get(
                "THINTENSOR_MXFP4_EXPERT_CACHE_BYTES",
                "0",
            )
        )
        self._mxfp4_expert_slice_cache: "OrderedDict[tuple[str, int], torch.Tensor]" = OrderedDict()
        self._mxfp4_expert_slice_cache_bytes = 0
        self._mxfp4_expert_materialize_buffers: dict[
            tuple[tuple[int, ...], torch.dtype], torch.Tensor
        ] = {}
        self._mxfp4_expert_materialize_bytes = 0
        self.residency_headroom_bytes = int(
            os.environ.get(
                "THINTENSOR_GPU_RESIDENCY_HEADROOM_BYTES",
                "0",
            )
        )
        self.lm_head_quantization_reserve_bytes = (
            self._lm_head_quantization_reserve_bytes()
        )
        self.persistent_pages = {
            page_id
            for page_id, page in self.page_specs.items()
            if (
                page_id != "model.embed_tokens.weight"
                or not self.cpu_embed
            ) and (
                page.get("kind") in {"embedding", "lm_head"}
                or page.get("layer") is None
            )
        }
        self.budget_resident_layers: list[int] = []
        self.budget_resident_expert_layers: list[int] = []
        self._select_budget_resident_layers()

        # Precompute page sets per layer for hot-path speedups
        model_layers = int(self.manifest["model"]["layers"])
        self._precomputed_layer_page_ids = {}
        self._precomputed_evictable_pages_by_layer = {}
        self._precomputed_active_layer_pages = {}
        self._precomputed_fused_parents = {}
        for layer in range(model_layers):
            self._precomputed_layer_page_ids[layer] = [
                page["id"]
                for page in self.page_specs.values()
                if page.get("kind") != "fused_physical"
                and page.get("layer") == layer
                and not _is_separate_expert_page(str(page["id"]))
                and str(page["id"]) not in self.streamed_packed_expert_pages
            ]
            self._precomputed_evictable_pages_by_layer[layer] = [
                page_id for page_id, page in self.page_specs.items()
                if page.get("layer") == layer
                and not _is_separate_expert_page(page_id)
                and page_id not in self.streamed_packed_expert_pages
            ]
            active_layer_pages = [
                page_id for page_id in self.page_specs.keys()
                if f"model.layers.{layer}." in page_id
            ]
            fused_parents = [
                self.page_specs[p]["fused_to"]
                for p in active_layer_pages
                if self.page_specs[p].get("fused_to") is not None
            ]
            self._precomputed_active_layer_pages[layer] = active_layer_pages
            self._precomputed_fused_parents[layer] = fused_parents

        # Prefetch tracking
        self._prefetch_events: Dict[str, torch.cuda.Event] = {}
        self._prefetch_in_progress: Set[str] = set()

        self.stats: dict[str, int | float] = {
            "cache_hits": 0,
            "cache_misses": 0,
            "prefetched_pages": 0,
            "evicted_pages": 0,
            "streamed_pages": 0,
            "fused_logical_pages": 0,
            "fused_physical_pages": 0,
            "aliased_pages": 0,
            "peak_resident_bytes": 0,
            "prefetch_hits": 0,
            "prefetch_waits": 0,
            "h2d_transfer_bytes": 0,
            "h2d_transfer_time_ms": 0.0,
            "selected_expert_h2d_transfer_bytes": 0,
            "selected_expert_h2d_transfer_time_ms": 0.0,
            "mxfp4_expert_slice_cache_hits": 0,
            "mxfp4_expert_slice_cache_misses": 0,
            "mxfp4_expert_slice_cache_evictions": 0,
            "mxfp4_expert_slice_cache_ooms": 0,
            "leaked_evicted_pages": 0,
        }

    def _page_resident_size(self, page: dict[str, Any]) -> int:
        source_dtype = map_dtype(page["dtype"])
        # Runtime-generated quantized pages are already in their execution
        # storage dtype. Do not estimate them as BF16 just because FP8 reports
        # as a floating-point dtype; doing so underfills the streaming cache by
        # several GiB on mixed-precision profiles.
        if (
            source_dtype in {torch.float8_e4m3fn, torch.int8, torch.uint8}
            or page.get("kind") == "scale"
            or page.get("quant_scheme") is not None
            or str(page.get("id", "")).endswith(".scale")
        ):
            target_dtype = source_dtype
        else:
            target_dtype = _target_dtype(source_dtype, self.dtype)
        elements = math.prod(int(dim) for dim in page["shape"])
        return elements * torch.empty((), dtype=target_dtype).element_size()

    def _select_budget_resident_layers(self) -> None:
        """Pin whole layers that fit beyond the streaming working set.

        A cyclic LRU cache is pathological for autoregressive decode: when the
        model is slightly larger than the budget, walking layers in order
        evicts the next layer just before it is needed and reloads the entire
        model every token. Keep a deterministic prefix resident and reserve
        enough space for the current layer plus prefetched layers.
        """
        # Identify active pages. Packed MXFP4 MoE tensors store every expert
        # for a layer in a few giant pages; under a tight budget the runtime
        # can stream only the routed expert slices instead of pinning or
        # prefetching those full pages.
        active_pages = set()
        packed_expert_pages = set()
        for page_id, page in self.page_specs.items():
            if page.get("kind") == "fused_physical":
                continue
            if self.cpu_embed and page_id == "model.embed_tokens.weight":
                continue
            if page_id in self._cpu_int4_pages or (
                page_id.endswith(".scale")
                and page_id[: -len(".scale")] in self._cpu_int4_pages
            ):
                continue
            page_layer = page.get("layer")
            if page_layer is not None:
                layer_idx = int(page_layer)
                # If this is an individual expert page for an INT4 layer, skip it (packed expert pages are used instead)
                if (
                    self.expert_int4
                    and layer_idx in self.selected_expert_int4_layers
                    and _is_separate_expert_page(page_id)
                ):
                    if not page_id.startswith("__runtime__"):
                        continue
            if _is_packed_expert_stream_page(page_id):
                packed_expert_pages.add(page_id)
            active_pages.add(page_id)

        full_active_pages = set(active_pages)
        full_active_bytes = sum(
            self._page_resident_size(self.page_specs[pid])
            for pid in full_active_pages
            if pid in self.page_specs
        )
        budget = self.vram_budget_bytes
        selection_budget = (
            max(
                0,
                budget
                - max(0, self.residency_headroom_bytes)
                - max(0, self.lm_head_quantization_reserve_bytes)
                - max(0, self.mxfp4_expert_slice_cache_limit_bytes),
            )
            if budget is not None
            else None
        )
        if (
            selection_budget is not None
            and full_active_bytes > selection_budget
            and packed_expert_pages
        ):
            base_active_pages = active_pages - packed_expert_pages
            base_active_bytes = sum(
                self._page_resident_size(self.page_specs[pid])
                for pid in base_active_pages
                if pid in self.page_specs
            )
            self.streamed_packed_expert_pages = set(packed_expert_pages)
            selected_expert_working_set = self._packed_expert_selected_working_set_bytes()
            packed_by_layer: dict[int, list[str]] = {}
            for page_id in packed_expert_pages:
                page = self.page_specs.get(page_id)
                if page is None or page.get("layer") is None:
                    continue
                packed_by_layer.setdefault(int(page["layer"]), []).append(page_id)

            pinned_packed_pages: set[str] = set()
            remaining_for_packed = max(
                0,
                selection_budget - base_active_bytes - selected_expert_working_set,
            )
            pinned_packed_bytes = 0
            for layer in sorted(packed_by_layer):
                layer_pages = packed_by_layer[layer]
                layer_bytes = sum(
                    self._page_resident_size(self.page_specs[pid])
                    for pid in layer_pages
                    if pid in self.page_specs
                )
                if pinned_packed_bytes + layer_bytes > remaining_for_packed:
                    break
                pinned_packed_pages.update(layer_pages)
                pinned_packed_bytes += layer_bytes
                self.budget_resident_expert_layers.append(layer)

            active_pages = base_active_pages | pinned_packed_pages
            self.streamed_packed_expert_pages = (
                packed_expert_pages - pinned_packed_pages
            )
            self.streamed_packed_expert_bytes = sum(
                self._page_resident_size(self.page_specs[pid])
                for pid in self.streamed_packed_expert_pages
                if pid in self.page_specs
            )
        else:
            self.streamed_packed_expert_pages = set()
            self.streamed_packed_expert_bytes = 0

        total_active_bytes = sum(
            self._page_resident_size(self.page_specs[pid])
            for pid in active_pages
            if pid in self.page_specs
        )
        selected_expert_working_set = (
            self._packed_expert_selected_working_set_bytes()
            if self.streamed_packed_expert_pages
            else 0
        )
        if selection_budget is None or full_active_bytes <= selection_budget:
            self.persistent_pages = active_pages
            self.budget_resident_layers = list(range(int(self.manifest["model"]["layers"])))
            if packed_expert_pages:
                self.budget_resident_expert_layers = list(
                    range(int(self.manifest["model"]["layers"]))
                )
            import sys
            budget_str = (
                f"{selection_budget / 1024**3:.2f} GiB"
                if selection_budget is not None
                else "Unlimited"
            )
            print(
                f"All weights fit within VRAM budget ({full_active_bytes / 1024**3:.2f} GiB <= {budget_str}); pinning all weights to VRAM.",
                file=sys.stderr,
            )
            return
        if (
            self.streamed_packed_expert_pages
            and selection_budget is not None
            and total_active_bytes + selected_expert_working_set <= selection_budget
        ):
            self.persistent_pages = active_pages
            self.budget_resident_layers = list(range(int(self.manifest["model"]["layers"])))
            import sys
            print(
                "Reusable weights and selected packed experts fit within VRAM budget "
                f"({total_active_bytes / 1024**3:.2f} GiB resident + "
                f"{selected_expert_working_set / 1024**3:.2f} GiB routed-expert "
                f"working set <= {selection_budget / 1024**3:.2f} GiB selection "
                f"budget, {self.residency_headroom_bytes / 1024**2:.0f} MiB "
                "headroom, "
                f"{self.lm_head_quantization_reserve_bytes / 1024**2:.0f} MiB "
                "head reserve); streaming "
                f"{self.streamed_packed_expert_bytes / 1024**3:.2f} GiB of "
                "packed MoE experts by selected slices.",
                file=sys.stderr,
            )
            return

        attn_pages_by_layer: dict[int, list[dict[str, Any]]] = {}
        mlp_pages_by_layer: dict[int, list[dict[str, Any]]] = {}
        for page in self.page_specs.values():
            layer = page.get("layer")
            if (
                layer is None
                or page.get("kind") == "fused_physical"
                or _is_separate_expert_page(str(page["id"]))
                or str(page["id"]) in self.streamed_packed_expert_pages
            ):
                continue
            if "attn" in str(page["id"]):
                attn_pages_by_layer.setdefault(int(layer), []).append(page)
            else:
                mlp_pages_by_layer.setdefault(int(layer), []).append(page)

        if not attn_pages_by_layer and not mlp_pages_by_layer:
            return

        global_bytes = sum(
            self._page_resident_size(self.page_specs[page_id])
            for page_id in self.persistent_pages
            if page_id in self.page_specs
            and self.page_specs[page_id].get("kind") != "fused_physical"
        )
        
        # Calculate working set bytes. We need to reserve enough room in the cache
        # to load the unpinned pages of the largest layer we might encounter.
        max_mlp_size = max(
            (sum(self._page_resident_size(page) for page in pages) for pages in mlp_pages_by_layer.values()),
            default=0
        )
        max_attn_size = max(
            (sum(self._page_resident_size(page) for page in pages) for pages in attn_pages_by_layer.values()),
            default=0
        )
        max_layer_size = max_mlp_size + max_attn_size

        expert_sizes = sorted(
            (
                self._page_resident_size(page)
                for page in self.page_specs.values()
                if _is_separate_expert_weight(str(page["id"]))
            ),
            reverse=True,
        )
        top_k = int(
            self.manifest.get("model", {}).get("num_experts_per_token") or 0
        )
        expert_working_set = sum(expert_sizes[: 3 * top_k]) * 2
        working_set_bytes = (
            max_layer_size * (1 + self.prefetch_distance)
            + expert_working_set
            + selected_expert_working_set
        )
        pin_budget = max(0, (selection_budget or 0) - global_bytes - working_set_bytes)

        pinned_bytes = 0
        # Pass 1: Pin attention projections first because they are small and highly reusable
        for layer in sorted(attn_pages_by_layer):
            size = sum(self._page_resident_size(page) for page in attn_pages_by_layer[layer])
            if pinned_bytes + size > pin_budget:
                break
            pinned_bytes += size
            self.persistent_pages.update(page["id"] for page in attn_pages_by_layer[layer])

        # Pass 2: Pin MLP projections with the remaining budget
        for layer in sorted(mlp_pages_by_layer):
            size = sum(self._page_resident_size(page) for page in mlp_pages_by_layer[layer])
            if pinned_bytes + size > pin_budget:
                break
            pinned_bytes += size
            self.budget_resident_layers.append(layer)
            self.persistent_pages.update(page["id"] for page in mlp_pages_by_layer[layer])

    def _cached_tensor(self, page_id: str) -> Optional[torch.Tensor]:
        dense_lowbit_base = (
            page_id[: -len(".scale")]
            if page_id.endswith(".scale")
            else page_id
        )
        if dense_lowbit_base in self._dense_lowbit_source_pages:
            if dense_lowbit_base not in self._dense_lowbit_cpu_cache:
                with self._quantization_lock:
                    if dense_lowbit_base not in self._dense_lowbit_cpu_cache:
                        cached = self._load_derived_dense_lowbit(
                            dense_lowbit_base
                        )
                        if cached is None:
                            source, _ = load_tensor_view(
                                self.archive, dense_lowbit_base
                            )
                            cached = quantize_dense_to_mxfp_lowbit_cpu(
                                source,
                                bits=self.dense_lowbit_bits,
                                chunk_rows=(256 if self.device.type == "cuda" else 64),
                                compute_device=(
                                    self.device
                                    if self.device.type == "cuda"
                                    else None
                                ),
                            )
                            self._write_derived_dense_lowbit(
                                dense_lowbit_base, cached
                            )
                        packed, scales = cached
                        self._dense_lowbit_cpu_cache[dense_lowbit_base] = (
                            packed,
                            scales,
                        )
                        self._weight_scales[dense_lowbit_base] = scales
            packed, scales = self._dense_lowbit_cpu_cache[dense_lowbit_base]
            return scales if page_id.endswith(".scale") else packed
        mxfp4_base = (
            page_id[: -len(".scale")]
            if page_id.endswith(".scale")
            else page_id
        )
        if mxfp4_base in self._mxfp4_source_pages:
            if mxfp4_base not in self._mxfp4_cpu_cache:
                with self._quantization_lock:
                    if mxfp4_base not in self._mxfp4_cpu_cache:
                        source, _ = load_tensor_view(
                            self.archive, mxfp4_base
                        )
                        packed, scales = (
                            ThinGpuCausalLMRuntime._quantize_mxfp4_rows(source)
                        )
                        self._mxfp4_cpu_cache[mxfp4_base] = (
                            packed, scales
                        )
                        self._weight_scales[mxfp4_base] = scales
            cached_mxfp4 = self._mxfp4_cpu_cache[mxfp4_base]
            return (
                cached_mxfp4[1]
                if page_id.endswith(".scale")
                else cached_mxfp4[0]
            )
        lowbit = self._packed_lowbit_by_page.get(page_id)
        if lowbit is not None:
            blocks_id, scales_id, bits = lowbit
            if blocks_id not in self._packed_lowbit_cache:
                with self._quantization_lock:
                    if blocks_id not in self._packed_lowbit_cache:
                        cached = self._load_derived_lowbit(blocks_id, scales_id)
                        if cached is None:
                            blocks, _ = load_tensor_view(self.archive, blocks_id)
                            scales, _ = load_tensor_view(self.archive, scales_id)
                            cached = quantize_mxfp4_to_lowbit_cpu(
                                blocks, scales, bits=bits
                            )
                            self._write_derived_lowbit(blocks_id, cached)
                        self._packed_lowbit_cache[blocks_id] = cached
            packed, scales = self._packed_lowbit_cache[blocks_id]
            return scales if page_id == scales_id else packed
        for layer, pack_ids in self._expert_int4_pack_page_ids.items():
            if page_id in pack_ids:
                self._build_expert_int4_pack(layer)
                return self._expert_int4_pack_cpu[page_id]
        int4_base = (
            page_id[: -len(".scale")]
            if page_id.endswith(".scale")
            else page_id
        )
        if int4_base in self._int4_source_pages:
            if int4_base not in self._int4_cpu_cache:
                with self._quantization_lock:
                    if int4_base not in self._int4_cpu_cache:
                        cached_int4 = self._load_derived_int4(int4_base)
                        if cached_int4 is None:
                            source = None
                            if (
                                self.cpu_store is not None
                                and int4_base in self.cpu_store.tensors
                            ):
                                source = self.cpu_store.tensors[int4_base]
                            if source is None:
                                source, _ = load_tensor_view(
                                    self.archive, int4_base
                                )
                            meta = self._int4_metadata_by_page.get(int4_base)
                            gsize = meta[2] if meta is not None else 32
                            cached_int4 = quantize_to_int4_cpu(
                                source,
                                group_size=gsize,
                                input_rms=self._dense_int4_input_rms.get(int4_base),
                                input_covariance32=self._load_int4_covariance(int4_base),
                            )
                            self._write_derived_int4(int4_base, cached_int4)
                        q_w, q_s = cached_int4
                        self._int4_cpu_cache[int4_base] = (q_w, q_s)
                        self._weight_scales[int4_base] = q_s
                        if self.cpu_store is not None:
                            self.cpu_store.put(int4_base, q_w)
                            self.cpu_store.put(int4_base + ".scale", q_s)
            cached_int4 = self._int4_cpu_cache[int4_base]
            return cached_int4[1] if page_id.endswith(".scale") else cached_int4[0]
        if page_id.endswith(".scale"):
            cached_int4 = self._int4_cpu_cache.get(page_id[: -len(".scale")])
            if cached_int4 is not None:
                return cached_int4[1]
        cached_int4 = self._int4_cpu_cache.get(page_id)
        if cached_int4 is not None:
            return cached_int4[0]
        if page_id.endswith(".scale"):
            cached = self._fp8_cpu_cache.get(page_id[: -len(".scale")])
            return cached[1] if cached is not None else None
        cached = self._fp8_cpu_cache.get(page_id)
        return cached[0] if cached is not None else None

    def _load_int4_covariance(self, page_id: str) -> Optional[torch.Tensor]:
        array_name = self._dense_int4_covariance_names.get(page_id)
        if array_name is None or self._dense_int4_covariance_path is None:
            return None
        with np.load(self._dense_int4_covariance_path) as arrays:
            return torch.from_numpy(arrays[array_name].copy())

    def _load_derived_int4(
        self, page_id: str
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        path = self._int4_cache_paths.get(page_id)
        if not self._derived_cache_enabled or path is None or not path.is_file():
            self.derived_cache_misses += 1
            return None
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            packed, scales = payload["packed"], payload["scales"]
            rows, cols, group_size = self._int4_metadata_by_page[page_id]
            groups = (cols + group_size - 1) // group_size
            expected_scale_shape = (
                (rows, groups, 2)
                if os.environ.get("THINTENSOR_AFFINE_INT4", "0") == "1" and rows < 65536
                else (rows, groups)
            )
            if (
                not isinstance(packed, torch.Tensor)
                or not isinstance(scales, torch.Tensor)
                or tuple(packed.shape) != (rows, cols // 2)
                or tuple(scales.shape) != expected_scale_shape
                or packed.dtype != torch.uint8
                or scales.dtype != torch.float16
            ):
                raise ValueError("derived INT4 tensor metadata mismatch")
        except (OSError, RuntimeError, KeyError, ValueError, TypeError):
            self.derived_cache_misses += 1
            try:
                path.unlink()
            except OSError:
                pass
            return None
        self.derived_cache_hits += 1
        self.derived_cache_bytes += path.stat().st_size
        return packed, scales

    def _write_derived_int4(
        self, page_id: str, tensors: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        path = self._int4_cache_paths.get(page_id)
        if not self._derived_cache_enabled or path is None or path.exists():
            return
        packed, scales = tensors
        required = _tensor_nbytes(packed) + _tensor_nbytes(scales)
        usage = shutil.disk_usage(path.parent.parent)
        if usage.free < required + 12 * 1024**3:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
            torch.save({"packed": packed, "scales": scales}, temporary)
            os.replace(temporary, path)
            self.derived_cache_writes += 1
            self.derived_cache_bytes += path.stat().st_size
        except OSError:
            try:
                temporary.unlink()
            except (OSError, UnboundLocalError):
                pass

    def _load_derived_dense_lowbit(
        self,
        page_id: str,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        path = self._dense_lowbit_cache_paths.get(page_id)
        if not self._derived_cache_enabled or path is None or not path.exists():
            self.derived_cache_misses += 1
            return None
        try:
            payload = torch.load(
                path,
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )
            packed = payload["packed"]
            scales = payload["scales"]
            rows, cols, bits = self._dense_lowbit_metadata_by_page[page_id]
            groups = cols // 32
            expected_packed = (rows, groups, 4 if bits == 1 else 8)
            if (
                not isinstance(packed, torch.Tensor)
                or not isinstance(scales, torch.Tensor)
                or tuple(packed.shape) != expected_packed
                or tuple(scales.shape) != (rows, groups)
                or packed.dtype != torch.uint8
                or scales.dtype != torch.uint8
            ):
                raise ValueError("derived dense low-bit tensor metadata mismatch")
        except (OSError, RuntimeError, KeyError, ValueError, TypeError):
            self.derived_cache_misses += 1
            try:
                path.unlink()
            except OSError:
                pass
            return None
        self.derived_cache_hits += 1
        self.derived_cache_bytes += path.stat().st_size
        return packed, scales

    def _write_derived_dense_lowbit(
        self,
        page_id: str,
        tensors: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        path = self._dense_lowbit_cache_paths.get(page_id)
        if not self._derived_cache_enabled or path is None or path.exists():
            return
        packed, scales = tensors
        required = _tensor_nbytes(packed) + _tensor_nbytes(scales)
        usage = shutil.disk_usage(path.parent.parent)
        if usage.used / max(1, usage.total) >= 0.88 or usage.free < required + 1024**3:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
            torch.save({"packed": packed, "scales": scales}, temporary)
            os.replace(temporary, path)
            self.derived_cache_writes += 1
            self.derived_cache_bytes += path.stat().st_size
        except OSError:
            try:
                temporary.unlink()
            except (OSError, UnboundLocalError):
                pass

    def _load_derived_lowbit(
        self,
        blocks_id: str,
        scales_id: str,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        path = self._packed_lowbit_cache_paths.get(blocks_id)
        if not self._derived_cache_enabled or path is None or not path.is_file():
            self.derived_cache_misses += 1
            return None
        try:
            payload = torch.load(
                path,
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )
            packed = payload["packed"]
            scales = payload["scales"]
            expected_packed = tuple(self.page_specs[blocks_id]["shape"])
            expected_scales = tuple(self.page_specs[scales_id]["shape"])
            if (
                not isinstance(packed, torch.Tensor)
                or not isinstance(scales, torch.Tensor)
                or tuple(packed.shape) != expected_packed
                or tuple(scales.shape) != expected_scales
                or packed.dtype != torch.uint8
                or scales.dtype != torch.uint8
            ):
                raise ValueError("derived low-bit tensor metadata mismatch")
        except (OSError, RuntimeError, KeyError, ValueError, TypeError):
            self.derived_cache_misses += 1
            try:
                path.unlink()
            except OSError:
                pass
            return None
        self.derived_cache_hits += 1
        self.derived_cache_bytes += path.stat().st_size
        return packed, scales

    def _write_derived_lowbit(
        self,
        blocks_id: str,
        tensors: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        path = self._packed_lowbit_cache_paths.get(blocks_id)
        if not self._derived_cache_enabled or path is None or path.exists():
            return
        packed, scales = tensors
        required = _tensor_nbytes(packed) + _tensor_nbytes(scales)
        usage = shutil.disk_usage(path.parent.parent)
        if usage.used / max(1, usage.total) >= 0.88 or usage.free < required + 1024**3:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
            torch.save({"packed": packed, "scales": scales}, temporary)
            os.replace(temporary, path)
            self.derived_cache_writes += 1
            self.derived_cache_bytes += path.stat().st_size
        except OSError:
            try:
                temporary.unlink()
            except (OSError, UnboundLocalError):
                pass

    def _build_expert_int4_pack(self, layer: int) -> None:
        ids = self._expert_int4_pack_page_ids[layer]
        if ids[0] in self._expert_int4_pack_cpu:
            return
        with self._quantization_lock:
            if ids[0] in self._expert_int4_pack_cpu:
                return
            shapes = [self.page_specs[page_id]["shape"] for page_id in ids]
            gate_up_q = torch.empty(tuple(shapes[0]), dtype=torch.uint8)
            gate_up_s = torch.empty(tuple(shapes[1]), dtype=torch.float32)
            down_q = torch.empty(tuple(shapes[2]), dtype=torch.uint8)
            down_s = torch.empty(tuple(shapes[3]), dtype=torch.float32)
            num_experts = int(shapes[0][0])
            gate_rows = int(shapes[0][1]) // 2
            for expert in range(num_experts):
                gate_id, up_id, down_id = _separate_expert_tensor_ids(
                    self, layer, expert
                )
                gate_source, _ = load_tensor_view(self.archive, gate_id)
                up_source, _ = load_tensor_view(self.archive, up_id)
                down_source, _ = load_tensor_view(self.archive, down_id)
                gate_q, gate_s = quantize_to_int4_cpu(
                    gate_source, group_size=self.expert_int4_group_size
                )
                up_q, up_s = quantize_to_int4_cpu(
                    up_source, group_size=self.expert_int4_group_size
                )
                down_q_one, down_s_one = quantize_to_int4_cpu(
                    down_source, group_size=self.expert_int4_group_size
                )
                gate_up_q[expert, :gate_rows].copy_(gate_q)
                gate_up_q[expert, gate_rows:].copy_(up_q)
                gate_up_s[expert, :gate_rows].copy_(gate_s)
                gate_up_s[expert, gate_rows:].copy_(up_s)
                down_q[expert].copy_(down_q_one)
                down_s[expert].copy_(down_s_one)
            self._expert_int4_pack_cpu.update(
                {
                    ids[0]: gate_up_q,
                    ids[1]: gate_up_s,
                    ids[2]: down_q,
                    ids[3]: down_s,
                }
            )

    def has_expert_int4_pack(self, layer: int) -> bool:
        return layer in self._expert_int4_pack_page_ids

    @property
    def is_fully_pinned(self) -> bool:
        active_pages = set()
        for page_id, page in self.page_specs.items():
            if page.get("kind") == "fused_physical":
                continue
            page_layer = page.get("layer")
            if page_layer is not None:
                layer_idx = int(page_layer)
                if (
                    self.expert_int4
                    and layer_idx in self.selected_expert_int4_layers
                    and _is_separate_expert_page(page_id)
                ):
                    if not page_id.startswith("__runtime__"):
                        continue
            active_pages.add(page_id)
        return active_pages.issubset(self.persistent_pages)

    def _mxfp4_selected_slice_bytes(self, page_id: str, selected: int) -> int:
        page = self.page_specs[page_id]
        shape = page.get("shape") or []
        experts = int(shape[0]) if shape else int(
            self.manifest.get("model", {}).get("num_local_experts") or 1
        )
        experts = max(1, experts)
        per_expert = math.ceil(self._page_resident_size(page) / experts)
        return per_expert * max(1, int(selected))

    def _lm_head_quantization_reserve_bytes(self) -> int:
        if not (self.lm_head_fp8 or self.lm_head_int4_group_size > 0):
            return 0
        head_page = self.page_specs.get("lm_head.weight")
        if head_page is None:
            head_page = self.page_specs.get("model.embed_tokens.weight")
        if head_page is None:
            return 0
        if head_page.get("quant_scheme") == "runtime_grouped_int4":
            return 0
        head_id = str(head_page.get("id") or "")
        if (
            head_id in self._fp8_cpu_cache
            or head_id in self._int4_cpu_cache
        ):
            return 0
        shape = head_page.get("shape") or []
        if len(shape) != 2:
            return 0
        rows = int(shape[0])
        cols = int(shape[1])
        if rows <= 0 or cols <= 0:
            return 0
        if self.lm_head_int4_group_size > 0:
            group_size = max(1, self.lm_head_int4_group_size)
            groups = (cols + group_size - 1) // group_size
            output_bytes = math.ceil(rows * cols / 2) + rows * groups * 4
        else:
            scale_blocks = (
                (cols + self.lm_head_fp8_scale_block - 1)
                // self.lm_head_fp8_scale_block
                if self.lm_head_fp8_scale_block > 0
                else 1
            )
            output_bytes = rows * cols + rows * scale_blocks * 4
        return int(output_bytes + 64 * 1024 * 1024)

    def _packed_expert_selected_working_set_bytes(self) -> int:
        if not self.streamed_packed_expert_pages:
            return 0
        top_k = int(
            self.manifest.get("model", {}).get("num_experts_per_token") or 1
        )
        top_k = max(1, top_k)
        by_layer: dict[int, dict[str, int]] = {}
        for page_id in self.streamed_packed_expert_pages:
            page = self.page_specs.get(page_id)
            if page is None:
                continue
            layer = page.get("layer")
            if layer is None:
                continue
            phase = "down" if "down_proj" in page_id else "gate_up"
            phases = by_layer.setdefault(int(layer), {})
            phases[phase] = phases.get(phase, 0) + self._mxfp4_selected_slice_bytes(
                page_id,
                top_k,
            )
        if not by_layer:
            return 0
        # Gate/up and down selected tensors can overlap briefly on the CUDA
        # stream. Reserve two phases so residency does not depend on allocator
        # timing.
        return max(max(phases.values()) for phases in by_layer.values()) * 2

    def expert_int4_pack(
        self, layer: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ids = self._expert_int4_pack_page_ids[layer]
        return tuple(self.tensor(page_id) for page_id in ids)  # type: ignore[return-value]

    def mxfp4_selected_expert_tensors(
        self,
        blocks_page_id: str,
        scales_page_id: str,
        expert_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return MXFP4 expert tensors for routed experts only.

        Packed MoE checkpoints store all experts for a layer in one large page.
        The compute kernel already supports selected experts, but loading the
        whole page in streaming mode makes decode PCIe-bound. If both pages are
        already resident, reuse them directly; otherwise copy just the routed
        expert slices and remap indices to the compact temporary tensors.
        """
        if blocks_page_id in self.tensors and scales_page_id in self.tensors:
            return (
                self.tensor(blocks_page_id),
                self.tensor(scales_page_id),
                expert_indices,
            )
        cpu_indices = expert_indices.detach().to(device="cpu", dtype=torch.long)
        unique_cpu, inverse_cpu = torch.unique(
            cpu_indices,
            sorted=True,
            return_inverse=True,
        )
        selected = int(unique_cpu.numel())
        incoming_bytes = self._mxfp4_selected_slice_bytes(
            blocks_page_id,
            selected,
        ) + self._mxfp4_selected_slice_bytes(scales_page_id, selected)
        self._evict_to_budget(
            f"{blocks_page_id}#selected_pair",
            incoming_bytes=incoming_bytes,
        )
        blocks = self._load_mxfp4_selected_slice(
            blocks_page_id,
            unique_cpu,
        )
        scales = self._load_mxfp4_selected_slice(
            scales_page_id,
            unique_cpu,
        )
        remapped = inverse_cpu.to(
            device=expert_indices.device,
            dtype=expert_indices.dtype,
            non_blocking=True,
        )
        return blocks, scales, remapped

    def packed_expert_bits(self, blocks_page_id: str) -> int:
        binding = self._packed_lowbit_by_page.get(blocks_page_id)
        return binding[2] if binding is not None else 4

    def _load_mxfp4_selected_slice(
        self,
        page_id: str,
        expert_indices: torch.Tensor,
    ) -> torch.Tensor:
        if page_id not in self.page_specs:
            raise KeyError(f"ThinTensor GPU page {page_id} is not present in manifest")
        # Router indices are computed on GPU. Streaming only selected experts
        # necessarily needs the tiny index vector on CPU to address host pages.
        cpu_indices = expert_indices.detach().to(device="cpu", dtype=torch.long)
        if (
            self.mxfp4_expert_slice_cache_limit_bytes > 0
            and page_id in self.streamed_packed_expert_pages
        ):
            cached = self._load_mxfp4_selected_slice_cached(page_id, cpu_indices)
            if cached is not None:
                return cached
        if self.cpu_store is not None and page_id in self.cpu_store.tensors:
            source = self.cpu_store.tensors[page_id]
        else:
            source, _ = load_tensor_view(self.archive, page_id)
        selected = source.index_select(0, cpu_indices).contiguous()
        target_dtype = _target_dtype(selected.dtype, self.dtype)
        incoming_bytes = selected.numel() * torch.empty(
            (),
            dtype=target_dtype,
        ).element_size()
        self._evict_to_budget(
            f"{page_id}#selected",
            incoming_bytes=incoming_bytes,
        )
        gpu, metric = self._copy_mxfp4_selected_to_gpu(
            page_id,
            selected,
            target_dtype,
        )
        self._record_mxfp4_selected_metric(metric)
        if gpu.is_cuda:
            gpu.record_stream(torch.cuda.current_stream(gpu.device))
        return gpu

    def _load_mxfp4_selected_slice_cached(
        self,
        page_id: str,
        cpu_indices: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        requested = [int(value) for value in cpu_indices.tolist()]
        rows: list[Optional[torch.Tensor]] = []
        missing: list[int] = []
        for expert in requested:
            key = (page_id, expert)
            cached = self._mxfp4_expert_slice_cache.get(key)
            if cached is None:
                rows.append(None)
                missing.append(expert)
                self.stats["mxfp4_expert_slice_cache_misses"] = int(
                    self.stats.get("mxfp4_expert_slice_cache_misses", 0)
                ) + 1
                continue
            self._mxfp4_expert_slice_cache.move_to_end(key)
            cached.record_stream(torch.cuda.current_stream(cached.device))
            rows.append(cached)
            self.stats["mxfp4_expert_slice_cache_hits"] = int(
                self.stats.get("mxfp4_expert_slice_cache_hits", 0)
            ) + 1

        if missing:
            if self.cpu_store is not None and page_id in self.cpu_store.tensors:
                source = self.cpu_store.tensors[page_id]
            else:
                source, _ = load_tensor_view(self.archive, page_id)
            missing_cpu = torch.tensor(missing, dtype=torch.long)
            selected = source.index_select(0, missing_cpu).contiguous()
            target_dtype = _target_dtype(selected.dtype, self.dtype)
            incoming_bytes = selected.numel() * torch.empty(
                (),
                dtype=target_dtype,
            ).element_size()
            self._trim_mxfp4_expert_slice_cache(
                max(
                    0,
                    self.mxfp4_expert_slice_cache_limit_bytes - incoming_bytes,
                )
            )
            self._evict_to_budget(
                f"{page_id}#selected",
                incoming_bytes=incoming_bytes,
            )
            gpu, metric = self._copy_mxfp4_selected_to_gpu(
                page_id,
                selected,
                target_dtype,
            )
            self._record_mxfp4_selected_metric(metric)
            if gpu.is_cuda:
                gpu.record_stream(torch.cuda.current_stream(gpu.device))
            for offset, expert in enumerate(missing):
                row = gpu.narrow(0, offset, 1)
                cacheable = True
                try:
                    # A contiguous narrow can still retain the full selected
                    # batch allocation. Clone so cache accounting matches the
                    # storage that remains live after this call returns.
                    cached_row = row.clone()
                except torch.OutOfMemoryError:
                    self.stats["mxfp4_expert_slice_cache_ooms"] = int(
                        self.stats.get("mxfp4_expert_slice_cache_ooms", 0)
                    ) + 1
                    self._trim_mxfp4_expert_slice_cache(0)
                    if self.device.type == "cuda":
                        torch.cuda.empty_cache()
                    cached_row = row
                    cacheable = False
                if cacheable:
                    self._insert_mxfp4_expert_slice_cache(
                        (page_id, expert),
                        cached_row,
                    )
                for idx, requested_expert in enumerate(requested):
                    if requested_expert == expert and rows[idx] is None:
                        rows[idx] = cached_row
                        break

        if not rows or any(row is None for row in rows):
            return None
        materialized = [row for row in rows if row is not None]
        if len(materialized) == 1:
            return materialized[0]
        try:
            return self._materialize_mxfp4_cached_rows(materialized)
        except torch.OutOfMemoryError:
            self.stats["mxfp4_expert_slice_cache_ooms"] = int(
                self.stats.get("mxfp4_expert_slice_cache_ooms", 0)
            ) + 1
            self._trim_mxfp4_expert_slice_cache(0)
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            return None

    def _materialize_mxfp4_cached_rows(
        self,
        rows: list[torch.Tensor],
    ) -> torch.Tensor:
        shape = (len(rows), *tuple(int(dim) for dim in rows[0].shape[1:]))
        key = (shape, rows[0].dtype)
        output = self._mxfp4_expert_materialize_buffers.get(key)
        if output is None:
            output_bytes = math.prod(shape) * rows[0].element_size()
            self._evict_to_budget(
                "__runtime__.mxfp4_expert_materialize",
                incoming_bytes=output_bytes,
            )
            output = torch.empty(shape, device=self.device, dtype=rows[0].dtype)
            self._mxfp4_expert_materialize_buffers[key] = output
            self._mxfp4_expert_materialize_bytes += output_bytes
            self.external_resident_bytes += output_bytes
            self.resident_bytes += output_bytes
            self.stats["peak_resident_bytes"] = max(
                int(self.stats["peak_resident_bytes"]), self.resident_bytes
            )
        torch.cat(rows, dim=0, out=output)
        return output

    def _copy_mxfp4_selected_to_gpu(
        self,
        page_id: str,
        selected: torch.Tensor,
        target_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, PageLoadMetrics]:
        try:
            return self.loader._copy_cpu_tensor(
                f"{page_id}#selected",
                selected,
                target_dtype,
            )
        except RuntimeError as exc:
            message = str(exc).lower()
            if "out of memory" not in message and "cuda oom" not in message:
                raise
            if not self._mxfp4_expert_slice_cache:
                raise
            self.stats["mxfp4_expert_slice_cache_ooms"] = int(
                self.stats.get("mxfp4_expert_slice_cache_ooms", 0)
            ) + 1
            self._trim_mxfp4_expert_slice_cache(0)
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            return self.loader._copy_cpu_tensor(
                f"{page_id}#selected",
                selected,
                target_dtype,
            )

    def _record_mxfp4_selected_metric(self, metric: PageLoadMetrics) -> None:
        self.loader.metrics.append(metric)
        self.stats["h2d_transfer_bytes"] = int(
            self.stats.get("h2d_transfer_bytes", 0)
        ) + metric.gpu_transfer_bytes
        self.stats["h2d_transfer_time_ms"] = float(
            self.stats.get("h2d_transfer_time_ms", 0.0)
        ) + metric.gpu_transfer_s * 1000.0
        self.stats["selected_expert_h2d_transfer_bytes"] = int(
            self.stats.get("selected_expert_h2d_transfer_bytes", 0)
        ) + metric.gpu_transfer_bytes
        self.stats["selected_expert_h2d_transfer_time_ms"] = float(
            self.stats.get("selected_expert_h2d_transfer_time_ms", 0.0)
        ) + metric.gpu_transfer_s * 1000.0

    def _insert_mxfp4_expert_slice_cache(
        self,
        key: tuple[str, int],
        tensor: torch.Tensor,
    ) -> None:
        if self.mxfp4_expert_slice_cache_limit_bytes <= 0:
            return
        tensor_bytes = _tensor_nbytes(tensor)
        if tensor_bytes > self.mxfp4_expert_slice_cache_limit_bytes:
            return
        previous = self._mxfp4_expert_slice_cache.pop(key, None)
        if previous is not None:
            self._mxfp4_expert_slice_cache_bytes = max(
                0,
                self._mxfp4_expert_slice_cache_bytes - _tensor_nbytes(previous),
            )
        self._trim_mxfp4_expert_slice_cache(
            self.mxfp4_expert_slice_cache_limit_bytes - tensor_bytes
        )
        self._mxfp4_expert_slice_cache[key] = tensor
        self._mxfp4_expert_slice_cache_bytes += tensor_bytes

    def _trim_mxfp4_expert_slice_cache(self, max_bytes: int) -> None:
        while (
            self._mxfp4_expert_slice_cache
            and self._mxfp4_expert_slice_cache_bytes > max(0, int(max_bytes))
        ):
            _, victim = self._mxfp4_expert_slice_cache.popitem(last=False)
            # Cache hits record the consuming stream on each tensor. Deleting
            # the last Python reference is therefore sufficient: the CUDA
            # caching allocator defers storage reuse until queued consumers
            # finish. A stream-wide synchronize here serialized dozens of LRU
            # evictions per decoded token on routed MoE workloads.
            self._mxfp4_expert_slice_cache_bytes = max(
                0,
                self._mxfp4_expert_slice_cache_bytes - _tensor_nbytes(victim),
            )
            self.stats["mxfp4_expert_slice_cache_evictions"] = int(
                self.stats.get("mxfp4_expert_slice_cache_evictions", 0)
            ) + 1

    # Retained for downstream integrations that used the old private helper.
    def _fp8_cached_tensor(self, page_id: str) -> Optional[torch.Tensor]:
        return self._cached_tensor(page_id)

    def warm_start(self) -> None:
        # Dense low-bit packing can use CUDA to reduce setup time, but its
        # temporary float/candidate tensors must not compete with an almost
        # fully resident model. Validate or build every selected derived page
        # before loading any of those pages to the GPU, then discard the host
        # views so the residency pass reopens the durable cache one page at a
        # time. This also makes an interrupted warm start resume cleanly.
        if self._dense_lowbit_source_pages:
            prepared = 0
            for page_id in sorted(self._dense_lowbit_source_pages):
                self._cached_tensor(page_id)
                self._dense_lowbit_cpu_cache.pop(page_id, None)
                self._weight_scales.pop(page_id, None)
                prepared += 1
            self.stats["dense_lowbit_prebuilt_pages"] = prepared
            gc.collect()
            malloc_trim()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        if self._cpu_int4_pages:
            for page_id in sorted(self._cpu_int4_pages):
                self._ensure_cpu_int4_kernel_page(page_id)
            self.stats["cpu_int4_prebuilt_pages"] = len(
                self._cpu_int4_kernel_cache
            )
            gc.collect()
            malloc_trim()
        for page_id in sorted(self.persistent_pages):
            if page_id in self.page_specs and self.page_specs[page_id].get("kind") != "fused_physical":
                self.ensure(page_id, reason="warm")
        self.loader.finish()
        # Repacked pages that are now pinned on GPU can always be regenerated
        # from the mmap archive. Do not retain a second model-sized host copy.
        for blocks_id in list(self._packed_lowbit_cache):
            _, scales_id, _ = self._packed_lowbit_by_page[blocks_id]
            if blocks_id in self.tensors and scales_id in self.tensors:
                self._packed_lowbit_cache.pop(blocks_id, None)
        for cache in (
            self._dense_lowbit_cpu_cache,
            self._int4_cpu_cache,
            self._mxfp4_cpu_cache,
            self._fp8_cpu_cache,
        ):
            for page_id in list(cache):
                if page_id in self.tensors and page_id + ".scale" in self.tensors:
                    cache.pop(page_id, None)
        malloc_trim()

    def _ensure_cpu_int4_kernel_page(
        self,
        page_id: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cached = self._cpu_int4_kernel_cache.get(page_id)
        if cached is not None:
            return cached
        self._cached_tensor(page_id)
        q_w, q_s = self._int4_cpu_cache.pop(page_id)
        rows, cols, group_size = self._int4_metadata_by_page[page_id]
        if self._cpu_int4_residual_terms > 0:
            exact, _ = load_tensor_view(self.archive, page_id)
            self._cpu_int4_residual_sidecars[page_id] = (
                self._build_cpu_int4_residual_sidecar(
                    page_id,
                    exact,
                    q_w,
                    q_s,
                    terms=self._cpu_int4_residual_terms,
                )
            )
        unpacked = torch.empty((rows, cols), dtype=torch.int32)
        unpacked[:, 0::2].copy_((q_w & 15).to(torch.int32))
        unpacked[:, 1::2].copy_((q_w >> 4).to(torch.int32))
        packed = torch.ops.aten._convert_weight_to_int4pack_for_cpu(
            unpacked,
            1,
        )
        del unpacked, q_w
        if q_s.ndim == 3:
            scales = q_s[:, :, 0].transpose(0, 1).to(torch.bfloat16)
            minimum = q_s[:, :, 1].transpose(0, 1).to(torch.bfloat16)
            zeros = (minimum + 8.0 * scales).to(torch.bfloat16)
            del minimum
        else:
            scales = q_s.transpose(0, 1).to(torch.bfloat16)
            zeros = torch.zeros_like(scales)
        scale_zeros = torch.stack((scales, zeros), dim=2).contiguous()
        del q_s, scales, zeros
        self._weight_scales.pop(page_id, None)
        self._cpu_int4_kernel_cache[page_id] = (packed, scale_zeros)
        self._tensor_id_to_page_id[id(packed)] = page_id
        return packed, scale_zeros

    def _build_cpu_int4_residual_sidecar(
        self,
        page_id: str,
        source: torch.Tensor,
        packed: torch.Tensor,
        scales: torch.Tensor,
        *,
        terms: int,
        chunk_rows: int = 16,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Keep the largest calibrated INT4 errors as a compact GPU correction."""
        rows, cols = map(int, source.shape)
        terms = min(max(0, int(terms)), cols)
        values = torch.empty((rows, terms), dtype=torch.bfloat16)
        indices = torch.empty((rows, terms), dtype=torch.int16)
        input_rms = self._dense_int4_input_rms.get(page_id)
        for start in range(0, rows, chunk_rows):
            end = min(rows, start + chunk_rows)
            exact = source[start:end].float()
            packed_chunk = packed[start:end]
            codes = torch.empty((end - start, cols), dtype=torch.float32)
            codes[:, 0::2] = (packed_chunk & 0x0F).float()
            codes[:, 1::2] = (packed_chunk >> 4).float()
            group_size = cols // int(scales.shape[1])
            grouped = codes.reshape(end - start, -1, group_size)
            if scales.ndim == 3:
                scale_chunk = scales[start:end, :, 0].to(torch.bfloat16).float()
                minimum_chunk = scales[start:end, :, 1].to(torch.bfloat16).float()
                zero_chunk = (minimum_chunk + 8.0 * scale_chunk).to(
                    torch.bfloat16
                ).float()
                reconstructed = (
                    (grouped - 8.0) * scale_chunk[:, :, None]
                    + zero_chunk[:, :, None]
                )
            else:
                reconstructed = (
                    (grouped - 8.0) * scales[start:end].float()[:, :, None]
                )
            error = exact - reconstructed.reshape(end - start, cols)
            ranking = error.abs()
            if input_rms is not None:
                ranking = ranking * input_rms.float()[None, :]
            selected = torch.topk(ranking, k=terms, dim=1, sorted=False).indices
            values[start:end].copy_(torch.gather(error, 1, selected).to(torch.bfloat16))
            indices[start:end].copy_(selected.to(torch.int16))
        gpu_values = values.to(device=self.device, non_blocking=False).contiguous()
        gpu_indices = indices.to(device=self.device, non_blocking=False).contiguous()
        self.register_external_resident_bytes(
            _tensor_nbytes(gpu_values) + _tensor_nbytes(gpu_indices)
        )
        return gpu_values, gpu_indices

    def cpu_int4_matvec(
        self,
        page_id: str,
        x: torch.Tensor,
        out: torch.Tensor,
    ) -> torch.Tensor:
        packed, scale_zeros = self._ensure_cpu_int4_kernel_page(page_id)
        _, _, group_size = self._int4_metadata_by_page[page_id]
        started = time.perf_counter()
        x_cpu = x.detach().to(device="cpu", dtype=torch.bfloat16).reshape(1, -1)
        result = torch.ops.aten._weight_int4pack_mm_for_cpu(
            x_cpu,
            packed,
            group_size,
            scale_zeros,
        )
        out.copy_(result.reshape(-1).to(device=out.device), non_blocking=False)
        self.stats["cpu_int4_matvec_calls"] = int(
            self.stats.get("cpu_int4_matvec_calls", 0)
        ) + 1
        self.stats["cpu_int4_matvec_seconds"] = float(
            self.stats.get("cpu_int4_matvec_seconds", 0.0)
        ) + (time.perf_counter() - started)
        self.stats["cpu_int4_matvec_bytes"] = int(
            self.stats.get("cpu_int4_matvec_bytes", 0)
        ) + _tensor_nbytes(packed) + _tensor_nbytes(scale_zeros)
        return out

    def cpu_int4_multi_matvec(
        self,
        page_ids: tuple[str, ...],
        x: torch.Tensor,
        out: torch.Tensor,
    ) -> torch.Tensor:
        """Run shared-input CPU INT4 projections with one D2H synchronization."""
        if not page_ids:
            return out
        kernels = [self._ensure_cpu_int4_kernel_page(page_id) for page_id in page_ids]
        group_sizes = [self._int4_metadata_by_page[page_id][2] for page_id in page_ids]
        started = time.perf_counter()
        x_cpu = x.detach().to(device="cpu", dtype=torch.bfloat16).reshape(1, -1)
        offset = 0
        bytes_read = 0
        for (packed, scale_zeros), group_size in zip(kernels, group_sizes):
            result = torch.ops.aten._weight_int4pack_mm_for_cpu(
                x_cpu,
                packed,
                group_size,
                scale_zeros,
            ).reshape(-1)
            rows = int(result.numel())
            out[offset : offset + rows].copy_(
                result.to(device=out.device), non_blocking=False
            )
            offset += rows
            bytes_read += _tensor_nbytes(packed) + _tensor_nbytes(scale_zeros)
        self.stats["cpu_int4_matvec_calls"] = int(
            self.stats.get("cpu_int4_matvec_calls", 0)
        ) + len(page_ids)
        self.stats["cpu_int4_transfer_batches"] = int(
            self.stats.get("cpu_int4_transfer_batches", 0)
        ) + 1
        self.stats["cpu_int4_matvec_seconds"] = float(
            self.stats.get("cpu_int4_matvec_seconds", 0.0)
        ) + (time.perf_counter() - started)
        self.stats["cpu_int4_matvec_bytes"] = int(
            self.stats.get("cpu_int4_matvec_bytes", 0)
        ) + bytes_read
        return out

    def close(self) -> None:
        # mmap-backed CPU embedding rows export a buffer pointer through the
        # tensor view. Drop that view before closing the archive mapping.
        self._cpu_embed_source = None
        gc.collect()
        try:
            self.archive.close()
        except BufferError:
            if not self.cpu_embed:
                raise
            # PyTorch can retain an internal storage export briefly after the
            # last Python tensor reference is gone. The isolated runner exits
            # immediately after close(), which releases the mapping and fd.
            # Do not turn a completed benchmark into a false failure.
            pass

    def tensor(self, page_id: str) -> torch.Tensor:
        if page_id in self._cpu_int4_pages:
            return self._ensure_cpu_int4_kernel_page(page_id)[0]
        if self.cpu_embed and page_id == "model.embed_tokens.weight":
            if self._cpu_embed_meta is None:
                page = self.page_specs[page_id]
                self._cpu_embed_meta = torch.empty(
                    tuple(int(dim) for dim in page["shape"]),
                    device="meta",
                    dtype=_target_dtype(map_dtype(page["dtype"]), self.dtype),
                )
            return self._cpu_embed_meta
        tensor = self.ensure(page_id, reason="demand")
        if tensor.is_cuda:
            # Pages may be allocated on the prefetch stream and evicted from the
            # pool while kernels on the compute stream still consume them.
            # Tell the caching allocator about that use before returning the
            # tensor so storage cannot be recycled until the compute stream has
            # passed all queued work that references it.
            tensor.record_stream(torch.cuda.current_stream(tensor.device))
        return tensor

    def copy_cpu_embedding_row(
        self,
        token_id: int | torch.Tensor,
        out: torch.Tensor,
    ) -> torch.Tensor:
        if not self.cpu_embed:
            raise RuntimeError("CPU embedding row requested without cpu_embed")
        if self._cpu_embed_source is None:
            self._cpu_embed_source, _ = load_tensor_view(
                self.archive, "model.embed_tokens.weight"
            )
        index = int(token_id.item()) if isinstance(token_id, torch.Tensor) else int(token_id)
        if index < 0 or index >= int(self._cpu_embed_source.shape[0]):
            raise ValueError(f"token_id {index} outside embedding vocabulary")
        out.copy_(self._cpu_embed_source[index], non_blocking=False)
        return out

    def has_page(self, page_id: str) -> bool:
        return page_id in self.page_specs

    def ensure(self, page_id: str, reason: str = "demand") -> torch.Tensor:
        self._reap_retired_tensors()
        if page_id in self.tensors:
            self.tensors.move_to_end(page_id)
            if reason == "demand" and page_id in self._prefetch_in_progress:
                if self.loader.prefetch_stream is not None:
                    torch.cuda.current_stream().wait_stream(self.loader.prefetch_stream)
                self.stats["prefetch_hits"] = int(self.stats.get("prefetch_hits", 0)) + 1
                event = self._prefetch_events.get(page_id)
                if event is not None:
                    if not event.query():
                        self.stats["prefetch_waits"] = int(self.stats.get("prefetch_waits", 0)) + 1
                self._prefetch_in_progress.discard(page_id)

            self.stats["cache_hits"] = int(self.stats["cache_hits"]) + 1
            return self.tensors[page_id]

        if page_id not in self.page_specs:
            raise KeyError(f"ThinTensor GPU page {page_id} is not present in manifest")

        self.stats["cache_misses"] = int(self.stats["cache_misses"]) + 1
        if reason == "prefetch":
            self.stats["prefetched_pages"] = int(self.stats["prefetched_pages"]) + 1
        elif reason == "demand":
            self.stats["streamed_pages"] = int(self.stats["streamed_pages"]) + 1

        page = self.page_specs[page_id]
        # A budget-driven demotion can evict a completed prefetched page. Its
        # bookkeeping must not make the subsequent demand path index a tensor
        # that is no longer resident.
        if page_id not in self.tensors:
            self._prefetch_in_progress.discard(page_id)
            self._prefetch_events.pop(page_id, None)
        if page.get("kind") == "fused_physical":
            return self._ensure_raw_fused(page_id, reason)
        key = (page["checksum"], tuple(page["shape"]), page["dtype"])
        if key in self.alias_keys and self.alias_keys[key] in self.tensors:
            tensor = self.tensors[self.alias_keys[key]]
            self._insert_tensor(page_id, tensor, 0)
            self.stats["aliased_pages"] = int(self.stats["aliased_pages"]) + 1
            return tensor

        if page.get("fused_to") is not None:
            parent_id = page["fused_to"]
            raw = self._ensure_raw_fused(parent_id, reason)
            tensor = self._logical_view_from_fused(page, raw)
            owned = 0 if tensor.dtype == map_dtype(page["dtype"]) else _tensor_nbytes(tensor)
            self._insert_tensor(page_id, tensor, owned)
            self.alias_keys[key] = page_id
            self.stats["fused_logical_pages"] = int(self.stats["fused_logical_pages"]) + 1
        else:
            if reason == "demand" and page_id in self._prefetch_in_progress:
                if self.loader.prefetch_stream is not None:
                    torch.cuda.current_stream().wait_stream(self.loader.prefetch_stream)
                self.stats["prefetch_hits"] = int(self.stats.get("prefetch_hits", 0)) + 1
                event = self._prefetch_events.get(page_id)
                if event is not None:
                    if not event.query():
                        self.stats["prefetch_waits"] = int(self.stats.get("prefetch_waits", 0)) + 1
                self._prefetch_in_progress.discard(page_id)
                tensor = self.tensors[page_id]
            else:
                stream = self.loader.prefetch_stream if reason == "prefetch" else None
                self._evict_to_budget(page_id, incoming_bytes=self._page_resident_size(page))
                tensor = self.loader.load_tensor(page, stream=stream)
                self._insert_tensor(page_id, tensor, _tensor_nbytes(tensor))
                self.alias_keys[key] = page_id

        # Update stats
        if self.loader.metrics:
            last_metric = self.loader.metrics[-1]
            if last_metric.page_id == page_id or (page.get("fused_to") and last_metric.page_id == page.get("fused_to")):
                self.stats["h2d_transfer_bytes"] = int(self.stats.get("h2d_transfer_bytes", 0)) + last_metric.gpu_transfer_bytes
                self.stats["h2d_transfer_time_ms"] = float(self.stats.get("h2d_transfer_time_ms", 0.0)) + last_metric.gpu_transfer_s * 1000.0

        self._evict_to_budget(page_id)
        return tensor

    def prefetch_layer(self, layer: int) -> None:
        if self.prefetch_distance == 0:
            return
        for page_id in self._layer_page_ids(layer):
            if page_id not in self.tensors:
                self.ensure(page_id, reason="prefetch")

    def evict_completed_layer(self, layer: int, keep_lag: int = 0) -> None:
        cutoff = layer - keep_lag
        if cutoff < 0:
            return
        if hasattr(self, "_precomputed_evictable_pages_by_layer"):
            for l in range(cutoff + 1):
                for page_id in self._precomputed_evictable_pages_by_layer.get(l, []):
                    if page_id in self.tensors and page_id not in self.persistent_pages:
                        self.evict(page_id)
            return
        for page_id, page in list(self.page_specs.items()):
            if page.get("layer") is None or int(page.get("layer", -1)) > cutoff:
                continue
            # Expert reuse is input dependent, not layer sequential. Let the
            # bounded LRU retain hot experts across tokens.
            if _is_separate_expert_page(page_id):
                continue
            if page_id in self.streamed_packed_expert_pages:
                continue
            if page_id not in self.persistent_pages:
                self.evict(page_id)

    def evict(self, page_id: str) -> None:
        if page_id in self.persistent_pages:
            is_tiny_norm = "norm" in page_id.lower() or self.owned_bytes.get(page_id, 0) < 1 * 1024 * 1024
            if is_tiny_norm:
                return
        
        page = self.page_specs.get(page_id)
        if page is None:
            return
            
        t = self.tensors.get(page_id)
        if t is not None and t.is_cuda:
            current = torch.cuda.current_stream(t.device)
            if (
                page_id in self._prefetch_in_progress
                and self.loader.prefetch_stream is not None
            ):
                current.wait_stream(self.loader.prefetch_stream)
            retired_bytes = self.owned_bytes.get(page_id, 0)
            event = torch.cuda.Event()
            event.record(current)
            self._retired_tensors.append((event, t, retired_bytes))
            self._retired_tensor_bytes += retired_bytes

        if page.get("kind") == "fused_physical":
            for logical_id, logical in list(self.page_specs.items()):
                if logical.get("fused_to") == page_id:
                    self._remove_tensor(logical_id)
        self._remove_tensor(page_id)

        parent = page.get("fused_to")
        if parent is not None and parent not in self.persistent_pages:
            live_children = [
                logical_id
                for logical_id, logical in self.page_specs.items()
                if logical.get("fused_to") == parent and logical_id in self.tensors
            ]
            if not live_children:
                # The logical view owns no bytes but retains the parent's CUDA
                # storage. Retire the physical page with an event so its real
                # allocation remains budgeted until all queued consumers finish.
                self.evict(parent)

        if self.debug_stream_refs and t is not None:
            import sys
            ref_count = sys.getrefcount(t)
            if ref_count > 2:
                print(
                    f"[DEBUG-STREAM-REFS] Page '{page_id}' evicted but still has "
                    f"{ref_count - 2} strong references!",
                    file=sys.stderr,
                )
                self.stats["leaked_evicted_pages"] = int(self.stats.get("leaked_evicted_pages", 0)) + 1

    def replace_resident_tensor(self, page_id: str, tensor: torch.Tensor) -> None:
        if page_id not in self.tensors:
            raise KeyError(f"cannot replace non-resident page {page_id}")
        previous_owned = self.owned_bytes.get(page_id, 0)
        owned = _tensor_nbytes(tensor) if previous_owned > 0 else 0
        self.tensors[page_id] = tensor
        self.owned_bytes[page_id] = owned
        self.resident_bytes += owned - previous_owned
        self.stats["peak_resident_bytes"] = max(
            int(self.stats["peak_resident_bytes"]), self.resident_bytes
        )

    def drop_persistent_page(self, page_id: str) -> None:
        self.persistent_pages.discard(page_id)
        self.evict(page_id)

    def promote_streamed_packed_expert_layers(
        self,
        reserve_bytes: int = 64 * 1024 * 1024,
    ) -> list[int]:
        if self.vram_budget_bytes is None or not self.streamed_packed_expert_pages:
            return []
        by_layer: dict[int, list[str]] = {}
        for page_id in self.streamed_packed_expert_pages:
            page = self.page_specs.get(page_id)
            if page is None or page.get("layer") is None:
                continue
            by_layer.setdefault(int(page["layer"]), []).append(page_id)

        promoted: list[int] = []
        budget = max(
            0,
            self.vram_budget_bytes
            - max(
                0,
                int(reserve_bytes),
                self.residency_headroom_bytes,
            )
            - max(0, self.mxfp4_expert_slice_cache_limit_bytes),
        )
        for layer in sorted(by_layer):
            page_ids = sorted(by_layer[layer])
            layer_bytes = sum(
                self._page_resident_size(self.page_specs[pid])
                for pid in page_ids
                if pid in self.page_specs
            )
            if (
                self.resident_bytes
                + self._retired_tensor_bytes
                + layer_bytes
                > budget
            ):
                break
            loaded_page_ids: list[str] = []
            try:
                for page_id in page_ids:
                    self.persistent_pages.add(page_id)
                    self.ensure(page_id, reason="warm")
                    loaded_page_ids.append(page_id)
            except RuntimeError as exc:
                message = str(exc).lower()
                if "out of memory" not in message and "cuda oom" not in message:
                    raise
                self.packed_expert_promotion_ooms += 1
                for page_id in page_ids:
                    self.persistent_pages.discard(page_id)
                for page_id in loaded_page_ids:
                    self.evict(page_id)
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                break
            self.streamed_packed_expert_pages.difference_update(page_ids)
            self.streamed_packed_expert_bytes = max(
                0,
                self.streamed_packed_expert_bytes - layer_bytes,
            )
            self.budget_resident_expert_layers.append(layer)
            promoted.append(layer)
        if promoted:
            self.budget_resident_expert_layers = sorted(
                set(self.budget_resident_expert_layers)
            )
            self.promoted_packed_expert_layers.extend(promoted)
            self.loader.finish()
        return promoted

    def register_external_resident_bytes(self, byte_count: int) -> None:
        self.external_resident_bytes += byte_count
        self.resident_bytes += byte_count
        self.stats["peak_resident_bytes"] = max(
            int(self.stats["peak_resident_bytes"]), self.resident_bytes
        )

    def telemetry(self) -> dict[str, Any]:
        self.loader._reap_completed_staging()
        self._reap_retired_tensors()
        metrics = self.loader.metrics
        
        p_bytes = 0
        p_on_gpu = 0
        for page_id in self.persistent_pages:
            spec = self.page_specs.get(page_id)
            if spec is not None:
                p_bytes += int(spec["size"])
            if page_id in self.tensors:
                p_on_gpu += 1

        prefetch_hits = int(self.stats.get("prefetch_hits", 0))
        prefetch_waits = int(self.stats.get("prefetch_waits", 0))
        prefetch_hits_no_wait = prefetch_hits - prefetch_waits
        prefetch_h2d_overlap_estimate = (
            float(prefetch_hits_no_wait) / max(1, prefetch_hits)
            if prefetch_hits > 0
            else 0.0
        )

        h2d_transfer_bytes = int(self.stats.get("h2d_transfer_bytes", 0))
        cache_hits = int(self.stats["cache_hits"])
        cache_misses = int(self.stats["cache_misses"])
        cache_hit_rate = float(cache_hits) / max(1, cache_hits + cache_misses)

        gpu_cache = {
            "resident_bytes": self.resident_bytes,
            "resident_and_expert_cache_bytes": (
                self.resident_bytes
                + self._retired_tensor_bytes
                + self._mxfp4_expert_slice_cache_bytes
            ),
            "retired_tensor_bytes": self._retired_tensor_bytes,
            "retired_tensor_count": len(self._retired_tensors),
            "peak_resident_bytes": int(self.stats["peak_resident_bytes"]),
            "resident_pages": len(self.tensors),
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "cache_hit_rate": cache_hit_rate,
            "evicted_pages": int(self.stats.get("evicted_pages", 0)),
            "h2d_transfer_bytes": h2d_transfer_bytes,
            "h2d_transfer_bytes_per_token": h2d_transfer_bytes / max(1, self.decode_steps_run),
            "h2d_transfer_time_ms": float(self.stats.get("h2d_transfer_time_ms", 0.0)),
            "selected_expert_h2d_transfer_bytes": int(
                self.stats.get("selected_expert_h2d_transfer_bytes", 0)
            ),
            "selected_expert_h2d_transfer_time_ms": float(
                self.stats.get("selected_expert_h2d_transfer_time_ms", 0.0)
            ),
            "mxfp4_expert_slice_cache_bytes": self._mxfp4_expert_slice_cache_bytes,
            "mxfp4_expert_slice_cache_limit_bytes": (
                self.mxfp4_expert_slice_cache_limit_bytes
            ),
            "mxfp4_expert_materialize_bytes": (
                self._mxfp4_expert_materialize_bytes
            ),
            "mxfp4_expert_slice_cache_entries": len(self._mxfp4_expert_slice_cache),
            "mxfp4_expert_slice_cache_hits": int(
                self.stats.get("mxfp4_expert_slice_cache_hits", 0)
            ),
            "mxfp4_expert_slice_cache_misses": int(
                self.stats.get("mxfp4_expert_slice_cache_misses", 0)
            ),
            "mxfp4_expert_slice_cache_evictions": int(
                self.stats.get("mxfp4_expert_slice_cache_evictions", 0)
            ),
            "mxfp4_expert_slice_cache_ooms": int(
                self.stats.get("mxfp4_expert_slice_cache_ooms", 0)
            ),
            "prefetch_hits": prefetch_hits,
            "prefetch_waits": prefetch_waits,
            "prefetch_h2d_overlap_estimate": prefetch_h2d_overlap_estimate,
        }

        cpu_store = {
            "resident_bytes": self.cpu_store.cpu_resident_bytes if self.cpu_store is not None else 0,
            "pinned_bytes": self.cpu_store.cpu_pinned_bytes if self.cpu_store is not None else 0,
            "page_count": self.cpu_store.cpu_page_count if self.cpu_store is not None else 0,
        }
        cpu_int4_packed_bytes = sum(
            _tensor_nbytes(packed) + _tensor_nbytes(scale_zeros)
            for packed, scale_zeros in self._cpu_int4_kernel_cache.values()
        )
        cpu_int4_residual_bytes = sum(
            _tensor_nbytes(values) + _tensor_nbytes(indices)
            for values, indices in self._cpu_int4_residual_sidecars.values()
        )

        p_on_cpu = 0
        if self.cpu_store is not None:
            p_on_cpu = len(self.persistent_pages) - p_on_gpu

        return {
            "cpu_offload_enabled": self.cpu_offload,
            "gpu_weight_budget_bytes": self.vram_budget_bytes or 0,
            "residency_headroom_bytes": self.residency_headroom_bytes,
            "lm_head_quantization_reserve_bytes": (
                self.lm_head_quantization_reserve_bytes
            ),
            "gpu_cache": gpu_cache,
            "cpu_store": cpu_store,
            "cpu_int4_kernel_pages": len(self._cpu_int4_kernel_cache),
            "cpu_int4_packed_bytes": cpu_int4_packed_bytes,
            "cpu_int4_residual_pages": len(self._cpu_int4_residual_sidecars),
            "cpu_int4_residual_gpu_bytes": cpu_int4_residual_bytes,
            "cpu_int4_matvec_calls": int(self.stats.get("cpu_int4_matvec_calls", 0)),
            "cpu_int4_transfer_batches": int(
                self.stats.get("cpu_int4_transfer_batches", 0)
            ),
            "cpu_int4_matvec_seconds": float(
                self.stats.get("cpu_int4_matvec_seconds", 0.0)
            ),
            "cpu_int4_matvec_bytes": int(
                self.stats.get("cpu_int4_matvec_bytes", 0)
            ),
            "persistent_pages": len(self.persistent_pages),
            "persistent_bytes": p_bytes,
            "budget_resident_layers": self.budget_resident_layers,
            "budget_resident_expert_layers": self.budget_resident_expert_layers,
            "promoted_packed_expert_layers": self.promoted_packed_expert_layers,
            "packed_expert_promotion_ooms": self.packed_expert_promotion_ooms,
            "streamed_packed_expert_pages": len(self.streamed_packed_expert_pages),
            "streamed_packed_expert_bytes": self.streamed_packed_expert_bytes,
            "persistent_pages_on_gpu": p_on_gpu,
            "persistent_pages_on_cpu": p_on_cpu,
            "cpu_offload_enabled_flag": self.cpu_offload,
            "cpu_resident_bytes": self.cpu_store.cpu_resident_bytes if self.cpu_store is not None else 0,
            "cpu_pinned_bytes": self.cpu_store.cpu_pinned_bytes if self.cpu_store is not None else 0,
            "cpu_page_count": self.cpu_store.cpu_page_count if self.cpu_store is not None else 0,
            "cpu_page_source": self.cpu_store.cpu_page_source if self.cpu_store is not None else "mmap",
            "cpu_cache_max_bytes": self.cpu_store.max_bytes if self.cpu_store is not None else 0,
            "cpu_cache_hits": self.cpu_store.cache_hits if self.cpu_store is not None else 0,
            "cpu_cache_misses": self.cpu_store.cache_misses if self.cpu_store is not None else 0,
            "cpu_cache_evictions": self.cpu_store.cache_evictions if self.cpu_store is not None else 0,
            "derived_cache_enabled": self._derived_cache_enabled,
            "derived_cache_hits": self.derived_cache_hits,
            "derived_cache_misses": self.derived_cache_misses,
            "derived_cache_writes": self.derived_cache_writes,
            "derived_cache_bytes": self.derived_cache_bytes,
            
            # Old fields for compatibility
            "resident_pages": len(self.tensors),
            "resident_bytes": self.resident_bytes,
            "external_resident_bytes": self.external_resident_bytes,
            "gpu_transfer_bytes": sum(metric.gpu_transfer_bytes for metric in metrics),
            "disk_read_s": sum(metric.disk_read_s for metric in metrics),
            "cpu_stage_s": sum(metric.cpu_stage_s for metric in metrics),
            "gpu_transfer_s": sum(metric.gpu_transfer_s for metric in metrics),
            "minor_page_faults": sum(metric.minor_page_faults for metric in metrics),
            "major_page_faults": sum(metric.major_page_faults for metric in metrics),
            "leaked_evicted_pages": int(self.stats.get("leaked_evicted_pages", 0)),
        }

    @property
    def resident_weight_bytes(self) -> int:
        return self.resident_bytes

    def _ensure_raw_fused(self, page_id: str, reason: str) -> torch.Tensor:
        if page_id in self.tensors:
            self.tensors.move_to_end(page_id)
            if reason == "demand" and page_id in self._prefetch_in_progress:
                if self.loader.prefetch_stream is not None:
                    torch.cuda.current_stream().wait_stream(self.loader.prefetch_stream)
                self.stats["prefetch_hits"] = int(self.stats.get("prefetch_hits", 0)) + 1
                event = self._prefetch_events.get(page_id)
                if event is not None:
                    if not event.query():
                        self.stats["prefetch_waits"] = int(self.stats.get("prefetch_waits", 0)) + 1
                self._prefetch_in_progress.discard(page_id)
            return self.tensors[page_id]

        if reason == "demand" and page_id in self._prefetch_in_progress:
            if self.loader.prefetch_stream is not None:
                torch.cuda.current_stream().wait_stream(self.loader.prefetch_stream)
            self.stats["prefetch_hits"] = int(self.stats.get("prefetch_hits", 0)) + 1
            event = self._prefetch_events.get(page_id)
            if event is not None:
                if not event.query():
                    self.stats["prefetch_waits"] = int(self.stats.get("prefetch_waits", 0)) + 1
            self._prefetch_in_progress.discard(page_id)
            raw = self.tensors[page_id]
        else:
            page = self.page_specs[page_id]
            stream = self.loader.prefetch_stream if reason == "prefetch" else None
            self._evict_to_budget(page_id, incoming_bytes=int(page["size"]))
            raw = self.loader.load_raw_page(page_id, int(page["size"]), stream=stream)
            self._insert_tensor(page_id, raw, _tensor_nbytes(raw))
            self.stats["fused_physical_pages"] = int(self.stats["fused_physical_pages"]) + 1
            if reason == "prefetch":
                self.stats["prefetched_pages"] = int(self.stats["prefetched_pages"]) + 1
        
        if self.loader.metrics:
            last_metric = self.loader.metrics[-1]
            if last_metric.page_id == page_id:
                self.stats["h2d_transfer_bytes"] = int(self.stats.get("h2d_transfer_bytes", 0)) + last_metric.gpu_transfer_bytes
                self.stats["h2d_transfer_time_ms"] = float(self.stats.get("h2d_transfer_time_ms", 0.0)) + last_metric.gpu_transfer_s * 1000.0

        self._evict_to_budget(page_id)
        return raw

    def _logical_view_from_fused(self, page: dict[str, Any], raw: torch.Tensor) -> torch.Tensor:
        source_dtype = map_dtype(page["dtype"])
        offset = int(page.get("fused_offset", 0))
        size = int(page["size"])
        byte_slice = raw.narrow(0, offset, size)
        typed = byte_slice.view(source_dtype)
        tensor = typed.reshape(tuple(int(dim) for dim in page["shape"]))
        target_dtype = _target_dtype(source_dtype, self.dtype)
        if target_dtype != source_dtype:
            tensor = tensor.to(dtype=target_dtype)
        return tensor

    def _layer_page_ids(self, layer: int) -> list[str]:
        if hasattr(self, "_precomputed_layer_page_ids") and layer in self._precomputed_layer_page_ids:
            return self._precomputed_layer_page_ids[layer]
        return [
            page["id"]
            for page in self.page_specs.values()
            if page.get("kind") != "fused_physical"
            and page.get("layer") == layer
            and not _is_separate_expert_page(str(page["id"]))
            and str(page["id"]) not in self.streamed_packed_expert_pages
        ]

    def _insert_tensor(self, page_id: str, tensor: torch.Tensor, owned_bytes: int) -> None:
        self.tensors[page_id] = tensor
        self._tensor_id_to_page_id[id(tensor)] = page_id
        self.owned_bytes[page_id] = owned_bytes
        self.resident_bytes += owned_bytes
        self.stats["peak_resident_bytes"] = max(
            int(self.stats["peak_resident_bytes"]), self.resident_bytes
        )

    def _remove_tensor(self, page_id: str) -> None:
        if page_id not in self.tensors:
            return
        tensor = self.tensors[page_id]
        del self.tensors[page_id]
        if self._tensor_id_to_page_id.get(id(tensor)) == page_id:
            self._tensor_id_to_page_id.pop(id(tensor), None)
        self._page_use_events.pop(page_id, None)
        self.resident_bytes = max(0, self.resident_bytes - self.owned_bytes.pop(page_id, 0))
        self.stats["evicted_pages"] = int(self.stats["evicted_pages"]) + 1

    def _evict_to_budget(self, page_id: str, incoming_bytes: int = 0) -> None:
        self._reap_retired_tensors()
        if self.vram_budget_bytes is not None and self.vram_budget_bytes > 0:
            budget = max(
                0,
                self.vram_budget_bytes
                - max(0, self.residency_headroom_bytes)
                - max(0, incoming_bytes),
            )
            if (
                self._mxfp4_expert_slice_cache_bytes > 0
                and self.resident_bytes
                + self._retired_tensor_bytes
                + self._mxfp4_expert_slice_cache_bytes
                > budget
            ):
                self._trim_mxfp4_expert_slice_cache(
                    max(
                        0,
                        budget
                        - self.resident_bytes
                        - self._retired_tensor_bytes,
                    )
                )
            while (
                self.resident_bytes
                + self._retired_tensor_bytes
                + self._mxfp4_expert_slice_cache_bytes
                > budget
            ):
                victim = None
                for candidate in self.tensors.keys():
                    use_event = self._page_use_events.get(candidate)
                    if (
                        candidate not in self.persistent_pages
                        and candidate != page_id
                        and candidate not in self._active_pages
                        and (use_event is None or use_event.query())
                    ):
                        victim = candidate
                        break
                if victim is None:
                    for candidate in self.tensors.keys():
                        if (
                            candidate not in self.persistent_pages
                            and candidate != page_id
                            and candidate not in self._active_pages
                        ):
                            victim = candidate
                            use_event = self._page_use_events.get(candidate)
                            if use_event is not None:
                                use_event.synchronize()
                            break
                if victim is None:
                    if self._retired_tensors:
                        self._retired_tensors[0][0].synchronize()
                        self._reap_retired_tensors()
                        continue
                    # Runtime-created quantization metadata can be larger than
                    # its manifest estimate. Demote a protected decoder page
                    # instead of dead-ending warm start; global pages remain
                    # protected and the demoted page follows normal LRU demand
                    # streaming thereafter.
                    for candidate in reversed(tuple(self.tensors.keys())):
                        spec = self.page_specs.get(candidate, {})
                        if (
                            candidate in self.persistent_pages
                            and spec.get("layer") is not None
                            and candidate != page_id
                            and candidate not in self._active_pages
                        ):
                            self.persistent_pages.discard(candidate)
                            victim = candidate
                            break
                    if victim is not None:
                        self.evict(victim)
                        continue
                    raise RuntimeError(
                        "GPU weight budget is below the protected execution "
                        "working set; increase --gpu-memory-budget"
                    )
                self.evict(victim)

    def _reap_retired_tensors(self) -> None:
        while self._retired_tensors:
            event, _tensor, retired_bytes = self._retired_tensors[0]
            if not event.query():
                break
            self._retired_tensors.popleft()
            self._retired_tensor_bytes = max(
                0, self._retired_tensor_bytes - retired_bytes
            )


@dataclass
class KVPage:
    layer: int
    head_group: int
    token_start: int
    token_count: int
    dtype: str
    location: str
    bytes: int
    payload: Optional[torch.Tensor] = None
    key_payload: Optional[torch.Tensor] = None
    value_payload: Optional[torch.Tensor] = None


class LayerPlan:
    def __init__(
        self,
        layer: int,
        pool: Optional["ThinGpuPagePool"] = None,
        input_layernorm_weight = None,
        q_proj = None,
        k_proj = None,
        v_proj = None,
        o_proj = None,
        q_norm = None,
        k_norm = None,
        post_attention_layernorm_weight = None,
        gate_proj = None,
        up_proj = None,
        down_proj = None,
        qkv_fused = None,
        gate_up_fused = None,
    ) -> None:
        self.layer = layer
        self.pool = pool
        
        self._input_layernorm_weight = input_layernorm_weight
        self._q_proj = q_proj
        self._k_proj = k_proj
        self._v_proj = v_proj
        self._o_proj = o_proj
        self._q_norm = q_norm
        self._k_norm = k_norm
        self._post_attention_layernorm_weight = post_attention_layernorm_weight
        self._gate_proj = gate_proj
        self._up_proj = up_proj
        self._down_proj = down_proj
        self._qkv_fused = qkv_fused
        self._gate_up_fused = gate_up_fused

        # Suffix strings
        self.input_layernorm_weight_id = _layer_tensor(layer, "input_layernorm.weight")
        self.q_proj_id = _layer_tensor(layer, "self_attn.q_proj.weight")
        self.k_proj_id = _layer_tensor(layer, "self_attn.k_proj.weight")
        self.v_proj_id = _layer_tensor(layer, "self_attn.v_proj.weight")
        self.o_proj_id = _layer_tensor(layer, "self_attn.o_proj.weight")
        self.q_norm_id = _layer_tensor(layer, "self_attn.q_norm.weight")
        self.k_norm_id = _layer_tensor(layer, "self_attn.k_norm.weight")
        self.post_attention_layernorm_weight_id = _layer_tensor(layer, "post_attention_layernorm.weight")
        self.gate_proj_id = _layer_tensor(layer, "mlp.gate_proj.weight")
        self.up_proj_id = _layer_tensor(layer, "mlp.up_proj.weight")
        self.down_proj_id = _layer_tensor(layer, "mlp.down_proj.weight")
        self.qkv_fused_id = f"layer_{layer}_attn_qkv_fused"
        self.gate_up_fused_id = f"layer_{layer}_mlp_gate_up_fused"

    @property
    def input_layernorm_weight(self) -> torch.Tensor:
        if self.pool is not None:
            return self.pool.tensor(self.input_layernorm_weight_id)
        return self._input_layernorm_weight

    @input_layernorm_weight.setter
    def input_layernorm_weight(self, val: torch.Tensor) -> None:
        self._input_layernorm_weight = val

    @property
    def q_proj(self) -> torch.Tensor:
        if self.pool is not None:
            return self.pool.tensor(self.q_proj_id)
        return self._q_proj

    @q_proj.setter
    def q_proj(self, val: torch.Tensor) -> None:
        self._q_proj = val

    @property
    def k_proj(self) -> torch.Tensor:
        if self.pool is not None:
            return self.pool.tensor(self.k_proj_id)
        return self._k_proj

    @k_proj.setter
    def k_proj(self, val: torch.Tensor) -> None:
        self._k_proj = val

    @property
    def v_proj(self) -> torch.Tensor:
        if self.pool is not None:
            return self.pool.tensor(self.v_proj_id)
        return self._v_proj

    @v_proj.setter
    def v_proj(self, val: torch.Tensor) -> None:
        self._v_proj = val

    @property
    def o_proj(self) -> torch.Tensor:
        if self.pool is not None:
            return self.pool.tensor(self.o_proj_id)
        return self._o_proj

    @o_proj.setter
    def o_proj(self, val: torch.Tensor) -> None:
        self._o_proj = val

    @property
    def q_norm(self) -> Optional[torch.Tensor]:
        if self.pool is not None:
            if self.pool.has_page(self.q_norm_id):
                return self.pool.tensor(self.q_norm_id)
            return None
        return self._q_norm

    @q_norm.setter
    def q_norm(self, val: Optional[torch.Tensor]) -> None:
        self._q_norm = val

    @property
    def k_norm(self) -> Optional[torch.Tensor]:
        if self.pool is not None:
            if self.pool.has_page(self.k_norm_id):
                return self.pool.tensor(self.k_norm_id)
            return None
        return self._k_norm

    @k_norm.setter
    def k_norm(self, val: Optional[torch.Tensor]) -> None:
        self._k_norm = val

    @property
    def post_attention_layernorm_weight(self) -> torch.Tensor:
        if self.pool is not None:
            return self.pool.tensor(self.post_attention_layernorm_weight_id)
        return self._post_attention_layernorm_weight

    @post_attention_layernorm_weight.setter
    def post_attention_layernorm_weight(self, val: torch.Tensor) -> None:
        self._post_attention_layernorm_weight = val

    @property
    def gate_proj(self) -> torch.Tensor:
        if self.pool is not None:
            return self.pool.tensor(self.gate_proj_id)
        return self._gate_proj

    @gate_proj.setter
    def gate_proj(self, val: torch.Tensor) -> None:
        self._gate_proj = val

    @property
    def up_proj(self) -> torch.Tensor:
        if self.pool is not None:
            return self.pool.tensor(self.up_proj_id)
        return self._up_proj

    @up_proj.setter
    def up_proj(self, val: torch.Tensor) -> None:
        self._up_proj = val

    @property
    def down_proj(self) -> torch.Tensor:
        if self.pool is not None:
            return self.pool.tensor(self.down_proj_id)
        return self._down_proj

    @down_proj.setter
    def down_proj(self, val: torch.Tensor) -> None:
        self._down_proj = val

    @property
    def qkv_fused(self) -> Optional[torch.Tensor]:
        if self.pool is not None:
            if self.pool.has_page(self.qkv_fused_id):
                return self.pool.tensor(self.qkv_fused_id)
            return None
        return self._qkv_fused

    @qkv_fused.setter
    def qkv_fused(self, val: Optional[torch.Tensor]) -> None:
        self._qkv_fused = val

    @property
    def gate_up_fused(self) -> Optional[torch.Tensor]:
        if self.pool is not None:
            if self.pool.has_page(self.gate_up_fused_id):
                return self.pool.tensor(self.gate_up_fused_id)
            return None
        return self._gate_up_fused

    @gate_up_fused.setter
    def gate_up_fused(self, val: Optional[torch.Tensor]) -> None:
        self._gate_up_fused = val


class PagedKVCache:
    """Paged KV cache with exact layouts and explicit capacity accounting.

    ``bytes`` intentionally retains its historical meaning: active allocated
    payload capacity.  ``telemetry`` additionally reports used bytes and
    padding so callers cannot mistake page capacity for populated KV data.
    """

    LAYOUTS = {
        "head_token_interleaved",
        "token_head_interleaved",
        "head_token_separate",
        "token_head_separate",
    }
    EXACT_CODECS = {
        "none",
        "bf16",
        "bfloat16",
        "fp16",
        "float16",
        "high_precision",
    }

    def __init__(
        self,
        layers: int,
        kv_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
        block_size: int = 16,
        recent_window: int = 256,
        old_codec: str = "bf16",
        budget_bytes: Optional[int] = None,
        policy: str = "sink_recent_attention",
        sink_tokens: int = 4,
        offload_old_to_cpu: bool = False,
        layout: str = "head_token_interleaved",
        residency: str = "gpu_full",
        gpu_recent_tokens: int = 256,
        prefetch_pages: int = 0,
    ) -> None:
        normalized_codec = str(old_codec).lower()
        if normalized_codec not in self.EXACT_CODECS:
            raise ValueError(
                f"KV codec {old_codec!r} is not executable: lossy KV "
                "payload encoding/decoding is not implemented. Use BF16/FP16 "
                "or an exact CPU residency mode."
            )
        if layout not in self.LAYOUTS:
            raise ValueError(
                f"unsupported KV layout {layout!r}; expected one of "
                f"{sorted(self.LAYOUTS)}"
            )
        if residency not in {"gpu_full", "cpu_exact", "hybrid_recent"}:
            raise ValueError(
                "KV residency must be gpu_full, cpu_exact, or hybrid_recent"
            )
        if offload_old_to_cpu and residency == "gpu_full":
            residency = "hybrid_recent"
        self.layers = layers
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype
        self.block_size = max(1, block_size)
        self.recent_window = max(0, recent_window)
        self.old_codec = normalized_codec
        self.budget_bytes = budget_bytes
        self.policy = policy
        self.sink_tokens = max(0, sink_tokens)
        self.offload_old_to_cpu = offload_old_to_cpu
        self.layout = layout
        self.residency = residency
        self.gpu_recent_tokens = max(0, gpu_recent_tokens)
        self.prefetch_pages = max(0, prefetch_pages)
        self.blocks: list[KVPage] = []
        self.blocks_by_layer: dict[int, list[KVPage]] = {}
        self.current: dict[int, KVPage] = {}
        self.bytes = 0
        self.peak_bytes = 0
        self.compressed_blocks = 0
        self.offloaded_blocks = 0
        self.evicted_blocks = 0
        self.tokens_attended = 0
        self.read_bytes = 0
        self.read_bytes_per_token = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.transfer_bytes = 0
        self.h2d_transfer_bytes = 0
        self.d2h_transfer_bytes = 0
        self.transfer_staging_peak_bytes = 0
        self.free_pages: list[KVPage] = []
        self.pool_reuse_hits = 0
        self.pool_reuse_misses = 0
        self.pool_peak_bytes = 0
        self.layer_windows: dict[int, int] = {}

    def reconfigure_geometry(self, kv_heads: int, head_dim: int) -> None:
        if kv_heads == self.kv_heads and head_dim == self.head_dim:
            return
        if self.blocks or self.current:
            raise RuntimeError(
                "cannot change KV geometry after blocks were allocated: "
                f"cache={self.kv_heads}x{self.head_dim}, "
                f"runtime={kv_heads}x{head_dim}"
            )
        self.kv_heads = kv_heads
        self.head_dim = head_dim

    def set_layer_window(
        self,
        layer: int,
        window: int | None,
    ) -> None:
        if window is None:
            self.layer_windows.pop(layer, None)
        elif window <= 0:
            raise ValueError(f"KV layer window must be positive, got {window}")
        else:
            self.layer_windows[layer] = int(window)

    def reset(self, *, reuse_pages: bool = True) -> None:
        """Clear logical KV state, optionally retaining exact allocations."""
        if reuse_pages:
            for block in self.blocks:
                self._release_page(block)
        self.blocks.clear()
        self.blocks_by_layer.clear()
        self.current.clear()
        self.bytes = 0
        self.tokens_attended = 0
        self.read_bytes = 0
        self.read_bytes_per_token = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.transfer_bytes = 0
        self.h2d_transfer_bytes = 0
        self.d2h_transfer_bytes = 0
        if not reuse_pages:
            self.free_pages.clear()

    def append(self, layer: int, key: torch.Tensor, value: torch.Tensor, token_index: int) -> None:
        actual_kv_heads = int(key.numel()) // self.head_dim
        if actual_kv_heads > 0 and actual_kv_heads != self.kv_heads:
            if self.blocks:
                raise RuntimeError(
                    f"KV head count changed from {self.kv_heads} to {actual_kv_heads}"
                )
            self.kv_heads = actual_kv_heads
        block = self.current.get(layer)
        if block is None or block.token_count >= self.block_size:
            block = self._new_block(layer, token_index)
            self.current[layer] = block
            self.blocks.append(block)
            self.blocks_by_layer.setdefault(layer, []).append(block)
            self.bytes += block.bytes
            self.peak_bytes = max(self.peak_bytes, self.bytes)
        block.token_count += 1
        self._store_token(block, key, value, block.token_count - 1)
        self._retier(layer, token_index)
        self._evict_sliding_history(layer, token_index)
        self._enforce_budget(token_index)

    def history(
        self,
        layer: int,
        token_index: int,
        start_token: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return exact K/V for start_token..token_index on the cache device."""
        keys: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        for block in self.blocks_by_layer.get(layer, []):
            if block.token_start > token_index:
                break
            if not self._has_payload(block):
                raise RuntimeError(
                    "causal_kv requires readable exact KV pages; "
                    f"layer={layer} block_start={block.token_start} "
                    f"location={block.location} codec={block.dtype}"
                )
            count = min(
                block.token_count,
                token_index - block.token_start + 1,
            )
            begin = max(0, start_token - block.token_start)
            if count <= begin:
                continue
            if block.location == "gpu":
                key_page, value_page = self._page_views(block)
                key_page = key_page[:, begin:count, :]
                value_page = value_page[:, begin:count, :]
                self.cache_hits += 1
            elif block.location == "cpu":
                key_page, value_page = self._materialize_gpu_views(
                    block,
                    begin,
                    count,
                )
                self.cache_misses += 1
            else:
                raise RuntimeError(
                    f"unsupported KV page location {block.location!r}"
                )
            keys.append(key_page)
            values.append(value_page)
        if not keys:
            raise RuntimeError(
                f"no KV entries available for layer={layer}, token={token_index}"
            )
        key = torch.cat(keys, dim=1) if len(keys) > 1 else keys[0]
        value = torch.cat(values, dim=1) if len(values) > 1 else values[0]
        attended = int(key.shape[1])
        read_bytes = _tensor_nbytes(key) + _tensor_nbytes(value)
        self.tokens_attended = attended
        self.read_bytes += read_bytes
        self.read_bytes_per_token = read_bytes
        return key, value

    def telemetry(self) -> dict[str, Any]:
        gpu_blocks = sum(1 for block in self.blocks if block.location == "gpu")
        cpu_blocks = sum(1 for block in self.blocks if block.location == "cpu")
        compressed_blocks = sum(1 for block in self.blocks if block.dtype == self.old_codec)
        active_allocated_bytes = sum(block.bytes for block in self.blocks)
        pool_bytes = sum(block.bytes for block in self.free_pages)
        allocated_bytes = active_allocated_bytes + pool_bytes
        used_bytes = sum(self._used_block_bytes(block) for block in self.blocks)
        gpu_bytes = pool_bytes + sum(
            block.bytes for block in self.blocks if block.location == "gpu"
        )
        cpu_bytes = sum(
            block.bytes for block in self.blocks if block.location == "cpu"
        )
        wasted_bytes = max(0, allocated_bytes - used_bytes)
        actual_sequence_length = max(
            (
                block.token_start + block.token_count
                for block in self.blocks
            ),
            default=0,
        )
        reserved_by_layer = {
            layer: sum(self._block_capacity(block) for block in blocks)
            for layer, blocks in self.blocks_by_layer.items()
        }
        max_sequence_length_reserved = max(
            reserved_by_layer.values(),
            default=0,
        )
        bytes_per_token = (
            2 * self.layers * self.kv_heads * self.head_dim
            * torch.empty((), dtype=self.dtype).element_size()
        )
        bytes_per_layer_token = (
            2 * self.kv_heads * self.head_dim
            * torch.empty((), dtype=self.dtype).element_size()
        )
        bytes_per_head_token = (
            2 * self.head_dim
            * torch.empty((), dtype=self.dtype).element_size()
        )
        return {
            "kv_blocks": len(self.blocks),
            "kv_gpu_blocks": gpu_blocks,
            "kv_cpu_blocks": cpu_blocks,
            "kv_compressed_blocks": compressed_blocks,
            "kv_bytes": self.bytes,
            "kv_allocated_bytes": allocated_bytes,
            "kv_active_allocated_bytes": active_allocated_bytes,
            "kv_used_bytes": used_bytes,
            "kv_wasted_padded_bytes": wasted_bytes,
            "kv_fragmentation_ratio": (
                wasted_bytes / allocated_bytes if allocated_bytes else 0.0
            ),
            "kv_gpu_bytes": gpu_bytes,
            "kv_cpu_bytes": cpu_bytes,
            "kv_bytes_per_token": bytes_per_token,
            "kv_bytes_per_layer_per_token": bytes_per_layer_token,
            "kv_bytes_per_head_per_token": bytes_per_head_token,
            "kv_bytes_per_layer": (
                used_bytes // self.layers if self.layers else 0
            ),
            "kv_bytes_per_head": (
                used_bytes // (self.layers * self.kv_heads)
                if self.layers and self.kv_heads
                else 0
            ),
            "kv_actual_sequence_length": actual_sequence_length,
            "kv_max_sequence_length_reserved": max_sequence_length_reserved,
            "kv_peak_bytes": self.peak_bytes,
            "kv_evicted_blocks": self.evicted_blocks,
            "kv_offloaded_blocks": self.offloaded_blocks,
            "kv_policy": self.policy,
            "kv_residency": self.residency,
            "kv_gpu_recent_tokens": self.gpu_recent_tokens,
            "kv_prefetch_pages": self.prefetch_pages,
            "kv_recent_window": self.recent_window,
            "kv_old_codec": self.old_codec,
            "kv_block_size": self.block_size,
            "kv_layout": self.layout,
            "kv_page_table_location": "python_host_v0",
            "kv_tokens_attended": self.tokens_attended,
            "kv_cache_read_bytes": self.read_bytes,
            "kv_cache_read_bytes_per_token": self.read_bytes_per_token,
            "kv_transfer_bytes": self.transfer_bytes,
            "kv_h2d_transfer_bytes": self.h2d_transfer_bytes,
            "kv_d2h_transfer_bytes": self.d2h_transfer_bytes,
            "kv_transfer_staging_peak_bytes": self.transfer_staging_peak_bytes,
            "kv_async_h2d": (
                self.device.type == "cuda"
                and any(
                    block.location == "cpu"
                    and self._payload_is_pinned(block)
                    for block in self.blocks
                )
            ),
            "kv_async_d2h": False,
            "kv_cache_hits": self.cache_hits,
            "kv_cache_misses": self.cache_misses,
            "kv_pool_free_pages": len(self.free_pages),
            "kv_pool_bytes": pool_bytes,
            "kv_pool_peak_bytes": self.pool_peak_bytes,
            "kv_pool_reuse_hits": self.pool_reuse_hits,
            "kv_pool_reuse_misses": self.pool_reuse_misses,
        }

    def _new_block(self, layer: int, token_start: int) -> KVPage:
        if self.free_pages:
            block = self.free_pages.pop()
            self.pool_reuse_hits += 1
            block.layer = layer
            block.token_start = token_start
            block.token_count = 0
            block.location = "gpu"
            block.dtype = str(self.dtype).replace("torch.", "")
            return block
        self.pool_reuse_misses += 1
        payload: Optional[torch.Tensor] = None
        key_payload: Optional[torch.Tensor] = None
        value_payload: Optional[torch.Tensor] = None
        if self.layout == "head_token_interleaved":
            payload = torch.empty(
                (2, self.kv_heads, self.block_size, self.head_dim),
                dtype=self.dtype,
                device=self.device,
            )
        elif self.layout == "token_head_interleaved":
            payload = torch.empty(
                (self.block_size, 2, self.kv_heads, self.head_dim),
                dtype=self.dtype,
                device=self.device,
            )
        elif self.layout == "head_token_separate":
            shape = (self.kv_heads, self.block_size, self.head_dim)
            key_payload = torch.empty(
                shape, dtype=self.dtype, device=self.device
            )
            value_payload = torch.empty(
                shape, dtype=self.dtype, device=self.device
            )
        else:
            shape = (self.block_size, self.kv_heads, self.head_dim)
            key_payload = torch.empty(
                shape, dtype=self.dtype, device=self.device
            )
            value_payload = torch.empty(
                shape, dtype=self.dtype, device=self.device
            )
        allocated_bytes = sum(
            _tensor_nbytes(tensor)
            for tensor in (payload, key_payload, value_payload)
            if tensor is not None
        )
        return KVPage(
            layer=layer,
            head_group=0,
            token_start=token_start,
            token_count=0,
            dtype=str(self.dtype).replace("torch.", ""),
            location="gpu",
            bytes=allocated_bytes,
            payload=payload,
            key_payload=key_payload,
            value_payload=value_payload,
        )

    def _release_page(self, block: KVPage) -> None:
        if block.location != "gpu" or not self._has_payload(block):
            return
        block.token_count = 0
        self.free_pages.append(block)
        self.pool_peak_bytes = max(
            self.pool_peak_bytes,
            sum(page.bytes for page in self.free_pages),
        )

    def _store_token(
        self,
        block: KVPage,
        key: torch.Tensor,
        value: torch.Tensor,
        slot: int,
    ) -> None:
        key = key.reshape(self.kv_heads, self.head_dim)
        value = value.reshape(self.kv_heads, self.head_dim)
        if self.layout == "head_token_interleaved":
            assert block.payload is not None
            block.payload[0, :, slot, :].copy_(key)
            block.payload[1, :, slot, :].copy_(value)
        elif self.layout == "token_head_interleaved":
            assert block.payload is not None
            block.payload[slot, 0, :, :].copy_(key)
            block.payload[slot, 1, :, :].copy_(value)
        elif self.layout == "head_token_separate":
            assert block.key_payload is not None
            assert block.value_payload is not None
            block.key_payload[:, slot, :].copy_(key)
            block.value_payload[:, slot, :].copy_(value)
        else:
            assert block.key_payload is not None
            assert block.value_payload is not None
            block.key_payload[slot, :, :].copy_(key)
            block.value_payload[slot, :, :].copy_(value)

    def _page_views(
        self,
        block: KVPage,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.layout == "head_token_interleaved":
            assert block.payload is not None
            return block.payload[0], block.payload[1]
        if self.layout == "token_head_interleaved":
            assert block.payload is not None
            return (
                block.payload[:, 0].permute(1, 0, 2),
                block.payload[:, 1].permute(1, 0, 2),
            )
        assert block.key_payload is not None
        assert block.value_payload is not None
        if self.layout == "head_token_separate":
            return block.key_payload, block.value_payload
        return (
            block.key_payload.permute(1, 0, 2),
            block.value_payload.permute(1, 0, 2),
        )

    def _materialize_gpu_views(
        self,
        block: KVPage,
        begin: int,
        count: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key_cpu, value_cpu = self._page_views(block)
        key_cpu = key_cpu[:, begin:count, :]
        value_cpu = value_cpu[:, begin:count, :]
        key_payload = key_cpu.to(
            device=self.device,
            non_blocking=key_cpu.is_pinned(),
        )
        value_payload = value_cpu.to(
            device=self.device,
            non_blocking=value_cpu.is_pinned(),
        )
        transferred = _tensor_nbytes(key_payload) + _tensor_nbytes(value_payload)
        self.transfer_bytes += transferred
        self.h2d_transfer_bytes += transferred
        self.transfer_staging_peak_bytes = max(
            self.transfer_staging_peak_bytes,
            transferred,
        )
        return key_payload, value_payload

    @staticmethod
    def _payload_is_pinned(block: KVPage) -> bool:
        payloads = (
            (block.payload,)
            if block.payload is not None
            else (block.key_payload, block.value_payload)
        )
        return all(
            payload is not None and payload.is_pinned()
            for payload in payloads
        )

    def _offload_page(self, block: KVPage) -> None:
        if block.location != "gpu" or not self._has_payload(block):
            return

        def cpu_copy(source: torch.Tensor) -> torch.Tensor:
            target = torch.empty(
                tuple(source.shape),
                dtype=source.dtype,
                device="cpu",
                pin_memory=self.device.type == "cuda",
            )
            target.copy_(source, non_blocking=False)
            self.transfer_bytes += _tensor_nbytes(source)
            self.d2h_transfer_bytes += _tensor_nbytes(source)
            return target

        if block.payload is not None:
            block.payload = cpu_copy(block.payload)
        else:
            assert block.key_payload is not None
            assert block.value_payload is not None
            block.key_payload = cpu_copy(block.key_payload)
            block.value_payload = cpu_copy(block.value_payload)
        block.location = "cpu"
        self.offloaded_blocks += 1

    @staticmethod
    def _has_payload(block: KVPage) -> bool:
        return block.payload is not None or (
            block.key_payload is not None
            and block.value_payload is not None
        )

    def _block_capacity(self, block: KVPage) -> int:
        scalar_bytes = (
            2 * self.kv_heads * self.head_dim
            * torch.empty((), dtype=self.dtype).element_size()
        )
        return block.bytes // scalar_bytes if scalar_bytes else 0

    def _used_block_bytes(self, block: KVPage) -> int:
        if block.dtype == str(self.dtype).replace("torch.", ""):
            return (
                2 * self.kv_heads * block.token_count * self.head_dim
                * torch.empty((), dtype=self.dtype).element_size()
            )
        return min(block.bytes, self._compressed_block_bytes(block.token_count))

    def _evict_sliding_history(
        self,
        layer: int,
        token_index: int,
    ) -> None:
        window = self.layer_windows.get(layer)
        if window is None:
            return
        first_required = max(0, token_index - window + 1)
        layer_blocks = self.blocks_by_layer.get(layer, [])
        retained = []
        removed = []
        for block in layer_blocks:
            block_end = block.token_start + block.token_count
            if (
                block is not self.current.get(layer)
                and block_end <= first_required
            ):
                removed.append(block)
            else:
                retained.append(block)
        if not removed:
            return
        self.blocks_by_layer[layer] = retained
        removed_ids = {id(block) for block in removed}
        self.blocks = [
            block for block in self.blocks if id(block) not in removed_ids
        ]
        for block in removed:
            self.bytes -= block.bytes
            self.evicted_blocks += 1
            self._release_page(block)

    def _retier(self, layer: int, token_index: int) -> None:
        if self.residency in {"cpu_exact", "hybrid_recent"}:
            keep_start = (
                token_index + 1
                if self.residency == "cpu_exact"
                else max(0, token_index - self.gpu_recent_tokens + 1)
            )
            for block in self.blocks_by_layer.get(layer, []):
                if block is self.current.get(layer):
                    continue
                block_end = block.token_start + block.token_count
                if block.location == "gpu" and block_end <= keep_start:
                    self._offload_page(block)
            return
        # Exact BF16/FP16 KV mode: do not compress/drop payloads.
        if str(self.old_codec).lower() in {"none", "bf16", "bfloat16", "fp16", "float16", "high_precision"} and not self.offload_old_to_cpu:
            return
        keep_start = max(0, token_index - self.recent_window + 1)
        for block in self.blocks_by_layer.get(layer, []):
            block_end = block.token_start + block.token_count
            is_sink = block.token_start < self.sink_tokens
            if block_end >= keep_start:
                break
            if is_sink or block.dtype == self.old_codec:
                continue
            old_bytes = block.bytes
            block.dtype = self.old_codec
            if self.offload_old_to_cpu:
                block.location = "cpu"
                block.payload = None
                block.key_payload = None
                block.value_payload = None
                self.offloaded_blocks += 1
            else:
                block.payload = None
                block.key_payload = None
                block.value_payload = None
            block.bytes = self._compressed_block_bytes(block.token_count)
            self.bytes = self.bytes - old_bytes + block.bytes
            self.compressed_blocks += 1

    def _enforce_budget(self, token_index: int) -> None:
        if self.budget_bytes is None or self.budget_bytes <= 0:
            return
        while self.bytes > self.budget_bytes:
            victim = self._eviction_candidate(token_index)
            if victim is None:
                break
            self.blocks.remove(victim)
            layer_blocks = self.blocks_by_layer.get(victim.layer)
            if layer_blocks is not None and victim in layer_blocks:
                layer_blocks.remove(victim)
            self.bytes = max(0, self.bytes - victim.bytes)
            self.evicted_blocks += 1
            self._release_page(victim)

    def _eviction_candidate(self, token_index: int) -> Optional[KVPage]:
        keep_start = max(0, token_index - self.recent_window + 1)
        candidates = [
            block
            for block in self.blocks
            if block.token_start >= self.sink_tokens
            and block.token_start + block.token_count < keep_start
        ]
        return candidates[0] if candidates else None

    def _compressed_block_bytes(self, token_count: int) -> int:
        scalars = 2 * self.kv_heads * max(1, token_count) * self.head_dim
        return int(scalars * _codec_bytes(self.old_codec))


class LayerProfileEvents:
    def __init__(self) -> None:
        self.layer_start = torch.cuda.Event(enable_timing=True)
        self.attn_start = torch.cuda.Event(enable_timing=True)
        self.qkv_start = torch.cuda.Event(enable_timing=True)
        self.qkv_end = torch.cuda.Event(enable_timing=True)
        self.qk_start = torch.cuda.Event(enable_timing=True)
        self.qk_end = torch.cuda.Event(enable_timing=True)
        self.softmax_start = torch.cuda.Event(enable_timing=True)
        self.softmax_end = torch.cuda.Event(enable_timing=True)
        self.value_mix_start = torch.cuda.Event(enable_timing=True)
        self.value_mix_end = torch.cuda.Event(enable_timing=True)
        self.o_proj_start = torch.cuda.Event(enable_timing=True)
        self.o_proj_end = torch.cuda.Event(enable_timing=True)
        self.attn_end = torch.cuda.Event(enable_timing=True)
        self.mlp_start = torch.cuda.Event(enable_timing=True)
        self.gate_proj_start = torch.cuda.Event(enable_timing=True)
        self.gate_proj_end = torch.cuda.Event(enable_timing=True)
        self.up_proj_start = torch.cuda.Event(enable_timing=True)
        self.up_proj_end = torch.cuda.Event(enable_timing=True)
        self.silu_mul_start = torch.cuda.Event(enable_timing=True)
        self.silu_mul_end = torch.cuda.Event(enable_timing=True)
        self.fused_mlp_start = torch.cuda.Event(enable_timing=True)
        self.fused_mlp_end = torch.cuda.Event(enable_timing=True)
        self.down_proj_start = torch.cuda.Event(enable_timing=True)
        self.down_proj_end = torch.cuda.Event(enable_timing=True)
        self.mlp_end = torch.cuda.Event(enable_timing=True)
        self.layer_end = torch.cuda.Event(enable_timing=True)


class ThinGpuCausalLMRuntime:
    """Manual single-token decoder runtime over canonical ThinTensor roles."""

    def __init__(
        self,
        weights: ThinGpuWeights | ThinGpuPagePool,
        kv_cache: Optional[PagedKVCache] = None,
        prefetch_distance: int = 0,
        evict_completed_layers: bool = False,
        kernel_backend: str = "torch",
        persistent_buffers: bool = True,
        lm_head_fp8: bool = False,
        keep_bf16_lm_head: bool = False,
        fused_mlp: bool = False,
        fused_scaled_mlp: bool = False,
        fused_residual_norm: bool = False,
        fused_rope: bool = False,
        tuned_large_matvec: bool = False,
        split_k_down_proj: bool = False,
        cuda_graphs: bool = False,
        down_proj_fp8: bool = False,
        mlp_fp8: bool = False,
        gate_up_fp8: bool = False,
        qkv_fp8: bool = False,
        o_proj_fp8: bool = False,
        attn_proj_fp8: bool = False,
        fp8_layer_spec: Optional[str] = None,
        down_fp8_layer_spec: Optional[str] = None,
        qkv_fp8_layer_spec: Optional[str] = None,
        o_fp8_layer_spec: Optional[str] = None,
        attention_mode: str = "causal_kv",
        attention_backend: str = "torch",
        exact_hf_mode: bool = False,
        fp8_scale_block: int = 0,
        fp8_residual_terms: int = 0,
        fp8_residual_layers: Optional[str] = None,
        fp8_residual_ranking: str = "raw",
        fp8_residual_dtype: str = "bf16",
        lm_head_fp8_scale_block: int = 0,
        lm_head_int4_group_size: int = 0,
        lm_head_backend: Optional[str] = None,
        lm_head_argmax_mode: str = "torch",
        lm_head_topk_guard: int = 0,
        gate_up_backend: Optional[str] = None,
        down_proj_backend: Optional[str] = None,
        attn_proj_backend: Optional[str] = None,
        adaptive_body_int8_start_token: int = -1,
        body_int4_group_size: int = 0,
        mxfp4_gate_up_layers: Optional[str] = None,
        mxfp4_down_layers: Optional[str] = None,
        mxfp4_qkv_layers: Optional[str] = None,
        mxfp4_o_layers: Optional[str] = None,
        mxfp4_hadamard: bool = False,
        mxfp4_hadamard_size: int = 0,
        mxfp4_hadamard_seed: int = 0,
        mxfp4_residual_terms: int = 0,
        mxfp4_binary_residual: bool = False,
        mxfp4_int8_row_fraction: float = 0.0,
        mxfp4_row_postscale: bool = False,
        moe_top_k_limit: int = 0,
        moe_top_k_renorm: bool = True,
    ) -> None:
        self.weights = weights
        self._use_tensor_cache = (
            not evict_completed_layers
            and isinstance(weights, ThinGpuWeights)
        )
        self.exact_hf_mode = exact_hf_mode
        self._adaptive_fp8_start_token = int(
            adaptive_body_int8_start_token
        )
        self._adaptive_body_int8 = self._adaptive_fp8_start_token >= 0
        # Until a caller identifies the generation boundary, retain the
        # historical absolute-token behavior used by decode microbenchmarks.
        # Interactive and validation paths call begin_decode() after prefill
        # so prompt length cannot silently consume the exact-token budget.
        self._adaptive_switch_token_index = self._adaptive_fp8_start_token
        self.body_int4_group_size = int(body_int4_group_size)
        self._body_int4 = self.body_int4_group_size > 0
        model_layers = int(weights.manifest["model"]["layers"])
        self.mxfp4_gate_up_layers = (
            _parse_layer_selection(mxfp4_gate_up_layers, model_layers)
            if mxfp4_gate_up_layers is not None
            else set()
        )
        self.mxfp4_down_layers = (
            _parse_layer_selection(mxfp4_down_layers, model_layers)
            if mxfp4_down_layers is not None
            else set()
        )
        self.mxfp4_qkv_layers = (
            _parse_layer_selection(mxfp4_qkv_layers, model_layers)
            if mxfp4_qkv_layers is not None
            else set()
        )
        self.mxfp4_o_layers = (
            _parse_layer_selection(mxfp4_o_layers, model_layers)
            if mxfp4_o_layers is not None
            else set()
        )
        self._body_mxfp4 = any(
            (
                self.mxfp4_gate_up_layers,
                self.mxfp4_down_layers,
                self.mxfp4_qkv_layers,
                self.mxfp4_o_layers,
            )
        )
        self.mxfp4_hadamard_size = int(mxfp4_hadamard_size)
        if mxfp4_hadamard and self.mxfp4_hadamard_size == 0:
            self.mxfp4_hadamard_size = 32
        if self.mxfp4_hadamard_size and (
            self.mxfp4_hadamard_size < 2
            or self.mxfp4_hadamard_size > 2048
            or self.mxfp4_hadamard_size
            & (self.mxfp4_hadamard_size - 1)
        ):
            raise ValueError(
                "mxfp4_hadamard_size must be 0 or a power of two in [2, 2048]"
            )
        self.mxfp4_hadamard = self.mxfp4_hadamard_size > 0
        self.mxfp4_hadamard_seed = int(mxfp4_hadamard_seed)
        self.mxfp4_residual_terms = max(0, int(mxfp4_residual_terms))
        self.mxfp4_binary_residual = bool(mxfp4_binary_residual)
        self.mxfp4_int8_row_fraction = float(mxfp4_int8_row_fraction)
        if not 0.0 <= self.mxfp4_int8_row_fraction <= 1.0:
            raise ValueError("mxfp4_int8_row_fraction must be in [0, 1]")
        self.mxfp4_row_postscale = bool(mxfp4_row_postscale)
        if self._body_int4 and (
            self.body_int4_group_size & (self.body_int4_group_size - 1)
        ):
            raise ValueError("body_int4_group_size must be a power of two")
        if self._body_int4 and self._adaptive_body_int8:
            raise ValueError(
                "adaptive body INT8 and body INT4 are separate opt-in modes"
            )
        if self._body_int4 and self._body_mxfp4:
            raise ValueError("body INT4 and body MXFP4 cannot be combined")
        if self._adaptive_body_int8 and not isinstance(
            weights, ThinGpuWeights
        ):
            raise ValueError(
                "adaptive body INT8 currently requires all-resident weights"
            )
        if self._body_int4 and not isinstance(weights, ThinGpuWeights):
            raise ValueError("body INT4 currently requires all-resident weights")
        if self._body_mxfp4 and torch.cuda.get_device_capability() < (12, 0):
            raise ValueError("native body MXFP4 requires Blackwell (SM 12.0+)")
        if fp8_scale_block < 0 or (
            fp8_scale_block
            and fp8_scale_block & (fp8_scale_block - 1)
        ):
            raise ValueError("fp8_scale_block must be zero or a power of two")
        self.fp8_scale_block = fp8_scale_block
        if lm_head_fp8_scale_block < 0 or (
            lm_head_fp8_scale_block
            and lm_head_fp8_scale_block
            & (lm_head_fp8_scale_block - 1)
        ):
            raise ValueError(
                "lm_head_fp8_scale_block must be zero or a power of two"
            )
        self.lm_head_fp8_scale_block = lm_head_fp8_scale_block
        self.lm_head_int4_group_size = int(lm_head_int4_group_size)
        self.lm_head_int4_enabled = self.lm_head_int4_group_size > 0
        if self.lm_head_int4_enabled and (
            self.lm_head_int4_group_size
            & (self.lm_head_int4_group_size - 1)
        ):
            raise ValueError("lm_head_int4_group_size must be a power of two")
        self.lm_head_backend_override = lm_head_backend
        if lm_head_topk_guard < 0:
            raise ValueError("lm_head_topk_guard must be non-negative")
        self.lm_head_topk_guard = int(lm_head_topk_guard)
        if lm_head_argmax_mode not in {"torch", "triton_two_stage", "triton_persistent"}:
            raise ValueError(
                "lm_head_argmax_mode must be torch, triton_two_stage, or triton_persistent"
            )
        self.lm_head_argmax_mode = lm_head_argmax_mode
        self.gate_up_backend_override = gate_up_backend
        self.down_proj_backend_override = down_proj_backend
        self.attn_proj_backend_override = attn_proj_backend
        self.cuda_graphs_requested = cuda_graphs
        self.down_proj_fp8 = (
            down_proj_fp8
            or mlp_fp8
            or self._adaptive_body_int8
            or self._body_int4
            or bool(self.mxfp4_down_layers)
        )
        self.gate_up_fp8 = (
            gate_up_fp8
            or mlp_fp8
            or self._adaptive_body_int8
            or self._body_int4
            or bool(self.mxfp4_gate_up_layers)
        )
        self.qkv_fp8 = (
            qkv_fp8
            or attn_proj_fp8
            or self._adaptive_body_int8
            or self._body_int4
            or bool(self.mxfp4_qkv_layers)
        )
        self.o_proj_fp8 = (
            o_proj_fp8
            or attn_proj_fp8
            or self._adaptive_body_int8
            or self._body_int4
            or bool(self.mxfp4_o_layers)
        )
        if exact_hf_mode and (
            lm_head_fp8
            or self.lm_head_int4_enabled
            or down_proj_fp8
            or mlp_fp8
            or gate_up_fp8
            or qkv_fp8
            or o_proj_fp8
            or attn_proj_fp8
            or self._adaptive_body_int8
            or self._body_int4
            or self._body_mxfp4
        ):
            raise ValueError("exact_hf_mode cannot be combined with FP8 modes")
        self.fp8_layers = _parse_layer_selection(
            fp8_layer_spec,
            int(weights.manifest["model"]["layers"]),
        )
        self.down_fp8_layers = _parse_layer_selection(
            down_fp8_layer_spec
            if down_fp8_layer_spec is not None
            else fp8_layer_spec,
            int(weights.manifest["model"]["layers"]),
        )
        self.qkv_fp8_layers = _parse_layer_selection(
            qkv_fp8_layer_spec
            if qkv_fp8_layer_spec is not None
            else fp8_layer_spec,
            int(weights.manifest["model"]["layers"]),
        )
        self.o_fp8_layers = _parse_layer_selection(
            o_fp8_layer_spec
            if o_fp8_layer_spec is not None
            else fp8_layer_spec,
            int(weights.manifest["model"]["layers"]),
        )
        if self._adaptive_body_int8:
            all_layers = set(range(int(weights.manifest["model"]["layers"])))
            if down_fp8_layer_spec is None:
                self.down_fp8_layers = all_layers
            if qkv_fp8_layer_spec is None:
                self.qkv_fp8_layers = all_layers
            if o_fp8_layer_spec is None:
                self.o_fp8_layers = all_layers
        elif self._body_int4:
            all_layers = set(range(int(weights.manifest["model"]["layers"])))
            self.fp8_layers = all_layers
            self.down_fp8_layers = all_layers
            self.qkv_fp8_layers = all_layers
            self.o_fp8_layers = all_layers
        self.fused_mlp_enabled_flag = (fused_mlp or (
            os.environ.get("THINTENSOR_FUSED_MLP", "0") == "1"
        )) and not self.gate_up_fp8
        self.fused_scaled_mlp_enabled_flag = (
            fused_scaled_mlp
            or os.environ.get("THINTENSOR_FUSED_SCALED_MLP", "0") == "1"
        ) and self.gate_up_fp8
        self.fused_residual_norm_enabled = (
            fused_residual_norm
            or os.environ.get("THINTENSOR_FUSED_RESIDUAL_NORM", "0") == "1"
        )
        self.fused_rope_enabled = (
            fused_rope
            or os.environ.get("THINTENSOR_FUSED_ROPE", "0") == "1"
        )
        self.tuned_large_matvec_enabled = (
            tuned_large_matvec
            or os.environ.get("THINTENSOR_TUNED_LARGE_MATVEC", "0") == "1"
        )
        self.split_k_down_proj_enabled = (
            split_k_down_proj
            or os.environ.get("THINTENSOR_SPLIT_K_DOWN_PROJ", "0") == "1"
        )
        self._fused_mlp_fallbacks = 0
        self._profiler_enabled = False
        self._profile_steps = []
        self.model = weights.manifest["model"]
        self.descriptor = descriptor_from_manifest(weights.manifest)
        self.execution_plan = compile_execution_plan(
            self.descriptor,
            (page["id"] for page in weights.manifest.get("pages", ())),
        )
        if not self.execution_plan.supported:
            raise RuntimeError(
                "native operator plan is not executable: "
                + "; ".join(self.execution_plan.unsupported_reasons)
            )
        self.layer_operator_contracts = tuple(
            layer.operators for layer in self.execution_plan.layers
        )
        self.operator_contract = frozenset(
            self.descriptor.required_operators
        ).union(
            operator
            for operators in self.layer_operator_contracts
            for operator in operators
        )
        try:
            self.runtime_executor = select_runtime_executor(
                (layer.schema for layer in self.execution_plan.layers),
                self.operator_contract,
                (layer.operators for layer in self.execution_plan.layers),
            )
        except ValueError as exc:
            raise RuntimeError(
                f"native operator plan has no executable graph: {exc}"
            ) from exc
        bind_resident_plan = not isinstance(weights, ThinGpuPagePool) or (
            weights.is_fully_pinned
        )
        self._bound_layer_tensors: tuple[dict[str, torch.Tensor], ...] = ()
        if bind_resident_plan:
            bound_layers = []
            for layer_plan in self.execution_plan.layers:
                roles = {
                    **layer_plan.tensors,
                    **layer_plan.optional_tensors,
                }
                bound_layers.append(
                    {
                        role: weights.tensor(page_id)
                        for role, page_id in roles.items()
                    }
                )
            self._bound_layer_tensors = tuple(bound_layers)
        self.is_moe = self.descriptor.is_moe
        self.model_moe_top_k = int(self.descriptor.num_experts_per_token)
        self.moe_top_k_limit = max(0, int(moe_top_k_limit))
        self.moe_top_k_renorm = bool(moe_top_k_renorm)
        selected_tensorcore_override = os.environ.get(
            "THINTENSOR_MXFP4_SELECTED_TENSORCORE"
        )
        self.mxfp4_selected_tensorcore = (
            selected_tensorcore_override == "1"
            if selected_tensorcore_override is not None
            else (
                weights.device.type == "cuda"
                and torch.cuda.get_device_capability(weights.device) >= (12, 0)
            )
        )
        self.effective_moe_top_k = self.model_moe_top_k
        if self.is_moe and self.moe_top_k_limit > 0:
            self.effective_moe_top_k = max(
                1,
                min(self.model_moe_top_k, self.moe_top_k_limit),
            )
        required_operators = self.operator_contract
        self.is_qwen3_5 = "gated_delta_net" in required_operators
        self.is_gemma4 = bool(
            {"per_layer_embeddings", "shared_kv_attention"}
            & required_operators
        )
        self.uses_dual_post_norm = {
            "post_attention_norm",
            "post_feedforward_norm",
        }.issubset(required_operators)
        self.gemma4_triton_attention = (
            self.is_gemma4
            and os.environ.get(
                "THINTENSOR_GEMMA4_TRITON_ATTENTION", "1"
            )
            == "1"
        )
        self.gemma4_triton_mlp = (
            self.is_gemma4
            and (
                os.environ.get("THINTENSOR_GEMMA4_TRITON_MLP", "0") == "1"
                or self.gate_up_fp8
                or self.down_proj_fp8
            )
        )
        self.gemma4_triton_ple = (
            self.is_gemma4
            and os.environ.get("THINTENSOR_GEMMA4_TRITON_PLE", "0")
            == "1"
        )
        if attention_mode not in {"causal_kv", "current_only_smoke"}:
            raise ValueError(
                "attention_mode must be causal_kv or current_only_smoke"
            )
        self.attention_mode = attention_mode
        if attention_backend not in {
            "torch",
            "sdpa",
            "triton_fused",
            "triton_split",
        }:
            raise ValueError(
                "attention_backend must be torch, sdpa, triton_fused, or "
                "triton_split"
            )
        self.attention_backend = (
            "torch" if exact_hf_mode else attention_backend
        )
        not_equivalent_reasons: list[str] = []
        if attention_mode != "causal_kv":
            not_equivalent_reasons.append(
                "current_only_smoke attends only to the current value vector"
            )
        if self.is_moe and self.effective_moe_top_k != self.model_moe_top_k:
            not_equivalent_reasons.append(
                f"MoE router executes top-{self.effective_moe_top_k} of "
                f"top-{self.model_moe_top_k}"
            )
        self.not_hf_equivalent = bool(not_equivalent_reasons)
        self.reason_not_equivalent = (
            "; ".join(not_equivalent_reasons)
            if not_equivalent_reasons
            else None
        )
        if self.descriptor.activation not in {
            "silu",
            "swish",
            "gelu_pytorch_tanh",
        }:
            raise RuntimeError(
                "manual runtime does not support gated activation "
                f"archive activation is {self.descriptor.activation!r}"
            )
        if self.is_moe and any(
            (
                self.down_proj_fp8,
                self.gate_up_fp8,
            )
        ):
            raise RuntimeError(
                "dense MLP FP8 flags cannot be applied to sparse-MoE experts; "
                "attention QKV/O FP8 remains supported by page role"
            )
        self.layers = int(self.model["layers"])
        self.hidden_size = int(self.model["hidden_size"])
        self.manifest_heads = int(self.model["heads"])
        self.manifest_kv_heads = int(self.model["kv_heads"])
        manifest_head_dim_value = self.model.get("head_dim")
        self.manifest_head_dim = (
            int(manifest_head_dim_value)
            if manifest_head_dim_value is not None
            else None
        )
        self.heads = self.manifest_heads
        self.kv_heads = self.manifest_kv_heads
        self.norm_kind = self.descriptor.norm_kind
        self.norm_eps = float(self.descriptor.norm_eps)
        self.norm_weight_offset = float(
            self.descriptor.norm_weight_offset
        )
        self.embedding_scale = float(self.descriptor.embedding_scale)
        self.attention_logit_softcap = (
            self.descriptor.attention_logit_softcap
        )
        self.final_logit_softcap = self.descriptor.final_logit_softcap
        self.attention_scale = (
            float(self.descriptor.query_pre_attn_scalar) ** -0.5
            if self.descriptor.query_pre_attn_scalar is not None
            else None
        )
        # Retained for head RMSNorm and older telemetry/call sites.
        self.rms_norm_eps = self.norm_eps
        self.partial_rotary_factor = float(
            self.descriptor.partial_rotary_factor
        )
        if self.norm_kind == "layer_norm":
            self.fused_residual_norm_enabled = False
        self.device = weights.device
        self.kv_cache = kv_cache
        self.prefetch_distance = max(0, prefetch_distance)
        self.evict_completed_layers = evict_completed_layers
        self._fused_matrix_cache: dict[str, torch.Tensor] = {}
        self.kernel_backend_name = kernel_backend
        self.persistent_buffers = persistent_buffers
        self.kernel_backend = None
        self.use_triton_matvec = kernel_backend in {"triton", "triton-matvec"}
        self.use_triton_elementwise = kernel_backend == "triton"
        self._logits_buffer: Optional[torch.Tensor] = None
        self._lm_head_tensor: Optional[torch.Tensor] = None
        self._lm_head_scale: Optional[torch.Tensor] = None
        self._lm_head_shortlist_rows: Optional[torch.Tensor] = None
        self._lm_head_shortlist_logits: Optional[torch.Tensor] = None
        self._lm_head_shortlist_bias: Optional[torch.Tensor] = None
        self._lm_head_shortlist_tie_ids: Optional[torch.Tensor] = None
        self._lm_head_vocab_sentinel: Optional[torch.Tensor] = None
        self._embed_scale: Optional[torch.Tensor] = None
        self.lm_head_fp8_enabled = lm_head_fp8 or self.lm_head_int4_enabled or (
            os.environ.get("THINTENSOR_LM_HEAD_FP8", "0") == "1"
        )
        self.keep_bf16_lm_head = (
            keep_bf16_lm_head or self.lm_head_topk_guard > 0
        )
        self._bf16_lm_head_tensor: Optional[torch.Tensor] = None
        self.lm_head_fp8_bytes = 0
        self.lm_head_external_resident_bytes = 0
        self.lm_head_bf16_resident = False
        self.lm_head_bf16_bytes = 0
        self.lm_head_memory_saved_bytes = 0
        self.lm_head_net_extra_bytes = 0
        self.lm_head_fp8_extra_bytes = 0
        self._fp8_head_weight_page_owned = False
        self.lm_head_tied_to_embeddings = self.descriptor.tie_word_embeddings
        self.lm_head_fp8_separate_execution_head = False
        # All-resident speed-mode runtime fusions.
        # These duplicate some weights in VRAM, so they are only used for ThinGpuWeights,
        # not streaming ThinGpuPagePool.
        self._runtime_fused_qkv_cache: dict[int, torch.Tensor] = {}
        self._runtime_fused_gate_up_cache: dict[int, torch.Tensor] = {}
        self.runtime_fusion_enabled = self._can_runtime_fuse_weights()
        self.runtime_fusion_extra_bytes = 0
        self.autotune_enabled = (
            os.environ.get("THINTENSOR_DISABLE_AUTOTUNE", "0") != "1"
            and isinstance(weights, ThinGpuWeights)
            and self.device.type == "cuda"
        )
        self._matvec_backend_choices: dict[tuple[Any, ...], str] = {}
        self._matvec_choice_by_tensor_id: dict[int, str] = {}
        self.per_shape_backend_benchmarks: dict[str, dict[str, Any]] = {}
        self._cuda_graph = None
        self._static_token_id = None
        self._static_next_token = None
        self.slot_tensor = None
        self._cuda_graphs_enabled = False
        self._cuda_graph_capture_s = 0.0
        self._cuda_graph_error = None
        self._last_kv_tokens_attended = 0
        self._debug_layer: Optional[int] = None
        self._debug_all_layers = False
        self._debug_components: dict[str, torch.Tensor] = {}
        self._rope_cos_sin_cache: dict[
            int, tuple[torch.Tensor, torch.Tensor]
        ] = {}
        self._rope_inv_freq: Optional[torch.Tensor] = None
        self._rope_attention_scaling = 1.0
        self._rope_dynamic_seq_len: Optional[int] = None

        self.has_fused_qkv_projection = _has_page(
            self.weights,
            _layer_tensor(0, "self_attn.qkv_proj.weight"),
        )
        self.has_fused_gate_up_projection = _has_page(
            self.weights,
            _layer_tensor(0, "mlp.gate_up_proj.weight"),
        )
        if self.is_gemma4:
            q_shape = self.weights.tensor(
                _layer_tensor(0, "self_attn.q_proj.weight")
            ).shape
            k_shape = self.weights.tensor(
                _layer_tensor(0, "self_attn.k_proj.weight")
            ).shape
            self.q_dim = int(q_shape[0])
            self.kv_dim = int(k_shape[0])
        elif self.is_qwen3_5:
            full_layers = [
                layer
                for layer, layer_type in enumerate(
                    self.descriptor.layer_types
                )
                if layer_type == "full_attention"
            ]
            if not full_layers:
                raise RuntimeError(
                    "Qwen3.5 archive has no full_attention layer"
                )
            full_layer = full_layers[0]
            q_shape = self.weights.tensor(
                _layer_tensor(full_layer, "self_attn.q_proj.weight")
            ).shape
            k_shape = self.weights.tensor(
                _layer_tensor(full_layer, "self_attn.k_proj.weight")
            ).shape
            if int(q_shape[0]) % 2:
                raise RuntimeError(
                    "Qwen3.5 gated q_proj must have an even row count"
                )
            self.q_dim = int(q_shape[0]) // 2
            self.kv_dim = int(k_shape[0])
        elif self.has_fused_qkv_projection:
            self.q_dim = self.manifest_heads * int(
                self.manifest_head_dim
                or self.descriptor.head_dim
            )
            self.kv_dim = self.manifest_kv_heads * int(
                self.manifest_head_dim
                or self.descriptor.head_dim
            )
            fused_rows = int(
                self.weights.tensor(
                    _layer_tensor(0, "self_attn.qkv_proj.weight")
                ).shape[0]
            )
            if fused_rows != self.q_dim + 2 * self.kv_dim:
                raise RuntimeError(
                    "fused QKV rows do not match manifest geometry: "
                    f"{fused_rows} != {self.q_dim}+2*{self.kv_dim}"
                )
        else:
            q_shape = self.weights.tensor(
                _layer_tensor(0, "self_attn.q_proj.weight")
            ).shape
            k_shape = self.weights.tensor(
                _layer_tensor(0, "self_attn.k_proj.weight")
            ).shape
            self.q_dim = int(q_shape[0])
            self.kv_dim = int(k_shape[0])
        self.head_dim = self._infer_head_dim()
        self.heads = self.q_dim // self.head_dim
        self.kv_heads = self.kv_dim // self.head_dim
        if self.kv_cache is not None:
            self.kv_cache.reconfigure_geometry(self.kv_heads, self.head_dim)
            for layer in range(self.layers):
                self.kv_cache.set_layer_window(
                    layer,
                    self.descriptor.layer_attention_window(layer),
                )
        if self.is_moe:
            self.intermediate_size = int(
                self.descriptor.intermediate_size
            )
        elif self.has_fused_gate_up_projection:
            fused_gate_shape = self.weights.tensor(
                _layer_tensor(0, "mlp.gate_up_proj.weight")
            ).shape
            self.intermediate_size = int(fused_gate_shape[0]) // 2
        else:
            gate_shape = self.weights.tensor(
                _layer_tensor(0, "mlp.gate_proj.weight")
            ).shape
            self.intermediate_size = int(gate_shape[0])
        if kernel_backend in {"triton", "triton-matvec", "hybrid"} and self.device.type != "cuda":
            raise RuntimeError(f"{kernel_backend} backend requires CUDA device")
        if kernel_backend in {"triton", "triton-matvec", "hybrid", "auto"} and self.device.type == "cuda":
            try:
                from .triton_kernels import TritonDecodeBackend

                requested_dtype = getattr(self.weights, "dtype", None)
                dtype = (
                    requested_dtype
                    if requested_dtype is not None
                    else self.weights.tensor("model.norm.weight").dtype
                )
                self.kernel_backend = TritonDecodeBackend(
                    self.device,
                    dtype,
                    self.hidden_size,
                    self.q_dim,
                    self.kv_dim,
                    self.intermediate_size,
                    argmax_mode=self.lm_head_argmax_mode,
                )
                self.kernel_backend_name = kernel_backend if kernel_backend != "auto" else "hybrid"
            except Exception as exc:
                if kernel_backend in {"triton", "triton-matvec", "hybrid"}:
                    raise
                self.kernel_backend_name = f"torch_fallback:{type(exc).__name__}"

        self._embed_weight = self.weights.tensor("model.embed_tokens.weight")
        self._final_norm_weight = self.weights.tensor("model.norm.weight")
        self._qwen35_conv_states: dict[int, torch.Tensor] = {}
        self._qwen35_recurrent_states: dict[int, torch.Tensor] = {}
        self._qwen35_projection_buffer: Optional[torch.Tensor] = None
        self._qwen35_core_buffer: Optional[torch.Tensor] = None
        self._gemma4_key_history: dict[int, list[torch.Tensor]] = {}
        self._gemma4_value_history: dict[int, list[torch.Tensor]] = {}
        self._gemma4_shared_source: dict[str, int] = {}
        self._gemma4_qkv_buffer: Optional[torch.Tensor] = None
        self._gemma4_gate_up_buffer: Optional[torch.Tensor] = None
        self._gemma4_mlp_buffer: Optional[torch.Tensor] = None
        self._gemma4_context_buffer: Optional[torch.Tensor] = None
        self._gemma4_ple_gate_buffer: Optional[torch.Tensor] = None
        self._gemma4_projection_buffer: Optional[torch.Tensor] = None
        if self.is_qwen3_5:
            conv_dim = (
                2
                * self.descriptor.linear_num_key_heads
                * self.descriptor.linear_key_head_dim
                + self.descriptor.linear_num_value_heads
                * self.descriptor.linear_value_head_dim
            )
            for layer, layer_type in enumerate(
                self.descriptor.layer_types
            ):
                if layer_type != "linear_attention":
                    continue
                self._qwen35_conv_states[layer] = torch.zeros(
                    (
                        conv_dim,
                        self.descriptor.linear_conv_kernel_dim,
                    ),
                    device=self.device,
                    dtype=self._final_norm_weight.dtype,
                )
                self._qwen35_recurrent_states[layer] = torch.zeros(
                    (
                        self.descriptor.linear_num_value_heads,
                        self.descriptor.linear_key_head_dim,
                        self.descriptor.linear_value_head_dim,
                    ),
                    device=self.device,
                    dtype=torch.float32,
                )
            self._qwen35_projection_buffer = torch.empty(
                max(
                    conv_dim
                    + self.descriptor.linear_num_value_heads
                    * self.descriptor.linear_value_head_dim
                    + 2 * self.descriptor.linear_num_value_heads,
                    self.q_dim * 2 + 2 * self.kv_dim,
                ),
                device=self.device,
                dtype=self._final_norm_weight.dtype,
            )
            self._qwen35_core_buffer = torch.empty(
                self.descriptor.linear_num_value_heads
                * self.descriptor.linear_value_head_dim,
                device=self.device,
                dtype=self._final_norm_weight.dtype,
            )
        if self.is_gemma4:
            first_shared = self.layers - self.descriptor.num_kv_shared_layers
            for layer in range(max(0, first_shared)):
                self._gemma4_key_history[layer] = []
                self._gemma4_value_history[layer] = []
                self._gemma4_shared_source[
                    self.descriptor.layer_types[layer]
                ] = layer
            max_intermediate = max(
                int(
                    self.weights.tensor(
                        _layer_tensor(
                            layer, "mlp.gate_proj.weight"
                        )
                    ).shape[0]
                )
                for layer in range(self.layers)
            )
            self._gemma4_qkv_buffer = torch.empty(
                self.heads * max(self.descriptor.head_dim, self.descriptor.global_head_dim)
                + 2 * self.descriptor.num_key_value_heads * max(self.descriptor.head_dim, self.descriptor.global_head_dim),
                device=self.device,
                dtype=self._final_norm_weight.dtype,
            )
            self._gemma4_gate_up_buffer = torch.empty(
                2 * max_intermediate,
                device=self.device,
                dtype=self._final_norm_weight.dtype,
            )
            self._gemma4_mlp_buffer = torch.empty(
                max_intermediate,
                device=self.device,
                dtype=self._final_norm_weight.dtype,
            )
            self._gemma4_context_buffer = torch.empty(
                self.layers
                * self.descriptor.hidden_size_per_layer_input,
                device=self.device,
                dtype=self._final_norm_weight.dtype,
            )
            self._gemma4_ple_gate_buffer = torch.empty(
                self.descriptor.hidden_size_per_layer_input,
                device=self.device,
                dtype=self._final_norm_weight.dtype,
            )
            self._gemma4_projection_buffer = torch.empty(
                self.hidden_size,
                device=self.device,
                dtype=self._final_norm_weight.dtype,
            )
        self._moe_gate_up_buffer: Optional[torch.Tensor] = None
        self._moe_down_buffer: Optional[torch.Tensor] = None
        if self.is_moe:
            top_k = self.effective_moe_top_k
            self._moe_gate_up_buffer = torch.empty(
                (top_k, 2 * self.intermediate_size),
                device=self.device,
                dtype=self._final_norm_weight.dtype,
            )
            self._moe_down_buffer = torch.empty(
                (top_k, self.hidden_size),
                device=self.device,
                dtype=self._final_norm_weight.dtype,
            )
            self._moe_activation_buffer = torch.empty(
                (top_k, self.intermediate_size),
                device=self.device,
                dtype=self._final_norm_weight.dtype,
            )
            self._moe_combined_buffer = torch.empty(
                (self.hidden_size,),
                device=self.device,
                dtype=self._final_norm_weight.dtype,
            )
            self._moe_router_logits_buffer = torch.empty(
                (self.descriptor.num_local_experts,),
                device=self.device,
                dtype=self._final_norm_weight.dtype,
            )
            self._moe_router_indices_buffer = torch.empty(
                (top_k,), device=self.device, dtype=torch.int32
            )
            self._moe_router_scores_buffer = torch.empty(
                (top_k,),
                device=self.device,
                dtype=self._final_norm_weight.dtype,
            )
        lm_head_source_shape: tuple[int, ...] | None = None
        if self.lm_head_fp8_enabled:
            if self.kernel_backend is None or not self.use_triton_matvec:
                raise RuntimeError("FP8 lm_head requires the Triton matvec backend")
            source_head = _optional_tensor(self.weights, "lm_head.weight")
            if source_head is None:
                source_head = self._embed_weight
            native_int4_meta = getattr(
                self.weights, "_int4_metadata_by_page", {}
            ).get("lm_head.weight")
            lm_head_source_shape = (
                (int(native_int4_meta[0]), int(native_int4_meta[1]))
                if native_int4_meta is not None
                else tuple(int(dim) for dim in source_head.shape)
            )
            source_bytes = math.prod(lm_head_source_shape) * 2
            cache_prefix = (
                "int4" if self.lm_head_int4_enabled else "fp8"
            )
            resident_head = getattr(
                self.weights, f"{cache_prefix}_lm_head_tensor", None
            )
            resident_scale = getattr(
                self.weights, f"{cache_prefix}_lm_head_scale", None
            )
            source_scale = None
            if self.lm_head_int4_enabled and source_head.dtype == torch.uint8:
                source_scale = _optional_tensor(
                    self.weights, "lm_head.weight.scale"
                )
            elif source_head.dtype == torch.float8_e4m3fn:
                source_scale = _optional_tensor(
                    self.weights,
                    (
                        "model.embed_tokens.weight.scale"
                        if source_head is self._embed_weight
                        or source_head.data_ptr() == self._embed_weight.data_ptr()
                        else "lm_head.weight.scale"
                    ),
                )
            if source_scale is not None:
                self._lm_head_tensor = source_head
                self._lm_head_scale = source_scale
                self._fp8_head_weight_page_owned = True
            elif resident_head is not None and resident_scale is not None:
                self._lm_head_tensor = resident_head
                self._lm_head_scale = resident_scale
            else:
                if self.lm_head_int4_enabled:
                    self._lm_head_tensor, self._lm_head_scale = (
                        self._quantize_int4_rows(
                            source_head,
                            group_size=self.lm_head_int4_group_size,
                        )
                    )
                else:
                    self._lm_head_tensor, self._lm_head_scale = (
                        self._quantize_fp8_rows(
                            source_head,
                            scale_block_size=self.lm_head_fp8_scale_block,
                        )
                    )
                setattr(
                    self.weights,
                    f"{cache_prefix}_lm_head_tensor",
                    self._lm_head_tensor,
                )
                setattr(
                    self.weights,
                    f"{cache_prefix}_lm_head_scale",
                    self._lm_head_scale,
                )
            self.lm_head_fp8_bytes = _tensor_nbytes(
                self._lm_head_tensor
            ) + _tensor_nbytes(self._lm_head_scale)
            self.lm_head_external_resident_bytes = 0
            if not self._fp8_head_weight_page_owned:
                self.lm_head_external_resident_bytes = _tensor_nbytes(
                    self._lm_head_scale
                )
                self.lm_head_external_resident_bytes += _tensor_nbytes(
                    self._lm_head_tensor
                )
            self._configure_fp8_head_residency(source_head)
            if (
                isinstance(self.weights, ThinGpuPagePool)
                and not getattr(
                    self.weights, "fp8_head_external_bytes_registered", False
                )
            ):
                external_bytes = 0
                if not self._fp8_head_weight_page_owned:
                    external_bytes = _tensor_nbytes(self._lm_head_scale)
                    external_bytes += _tensor_nbytes(self._lm_head_tensor)
                if external_bytes:
                    self.weights.register_external_resident_bytes(external_bytes)
                self.weights.fp8_head_external_bytes_registered = True
            if self.lm_head_fp8_separate_execution_head:
                self.lm_head_memory_saved_bytes = 0
                self.lm_head_net_extra_bytes = self.lm_head_fp8_bytes
            else:
                self.lm_head_memory_saved_bytes = max(
                    0, source_bytes - self.lm_head_fp8_bytes
                )
                self.lm_head_net_extra_bytes = (
                    self.lm_head_fp8_bytes
                    if self.lm_head_bf16_resident
                    else self.lm_head_fp8_bytes - source_bytes
                )
            self.lm_head_fp8_extra_bytes = max(0, self.lm_head_net_extra_bytes)

        if self._embed_weight.dtype == torch.float8_e4m3fn:
            embed_scale = _optional_tensor(
                self.weights,
                "model.embed_tokens.weight.scale",
            )
            if embed_scale is None:
                raise RuntimeError(
                    "FP8 embedding page is missing model.embed_tokens.weight.scale"
                )
            self._embed_scale = embed_scale
            if not _has_page(self.weights, "lm_head.weight"):
                self._lm_head_tensor = self._embed_weight
                self._lm_head_scale = embed_scale
                self._fp8_head_weight_page_owned = True
                self.lm_head_fp8_enabled = True
                self.lm_head_tied_to_embeddings = True
                self.lm_head_fp8_bytes = (
                    _tensor_nbytes(self._lm_head_tensor)
                    + _tensor_nbytes(self._lm_head_scale)
                )
                source_bytes = self._embed_weight.numel() * 2
                self.lm_head_memory_saved_bytes = max(
                    0,
                    source_bytes - self.lm_head_fp8_bytes,
                )

        if (
            "source_head" in locals()
            and not self.lm_head_bf16_resident
            and source_head is not self._embed_weight
        ):
            del source_head
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        if (
            isinstance(self.weights, ThinGpuPagePool)
            and self.weights.streamed_packed_expert_pages
            and os.environ.get("THINTENSOR_PROMOTE_PACKED_EXPERTS", "1") != "0"
        ):
            reserve_bytes = int(
                os.environ.get(
                    "THINTENSOR_PROMOTED_PACKED_EXPERT_RESERVE_BYTES",
                    str(64 * 1024 * 1024),
                )
            )
            self.weights.promote_streamed_packed_expert_layers(
                reserve_bytes=reserve_bytes,
            )

        self._weight_scales: Dict[int, torch.Tensor] = {}
        self._int4_metadata: Dict[int, tuple[int, int, int]] = {}
        self._mxfp4_metadata: Dict[int, tuple[int, int, int]] = {}
        self._dense_lowbit_metadata: Dict[int, tuple[int, int, int]] = {}
        self._mxfp4_hadamard_buffers: Dict[int, torch.Tensor] = {}
        self._mxfp4_hadamard_signs: Dict[int, torch.Tensor] = {}
        self._mxfp4_binary_residuals: Dict[
            int, tuple[torch.Tensor, torch.Tensor]
        ] = {}
        self._mxfp4_int8_row_overrides: Dict[
            int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ] = {}
        self._mxfp4_row_postscales: Dict[int, torch.Tensor] = {}
        if self.lm_head_int4_enabled:
            assert self._lm_head_tensor is not None
            if lm_head_source_shape is None:
                lm_head_source_shape = tuple(
                    int(dim) for dim in self._lm_head_tensor.shape
                )
            self._int4_metadata[id(self._lm_head_tensor)] = (
                int(lm_head_source_shape[0]),
                int(lm_head_source_shape[1]),
                self.lm_head_int4_group_size,
            )
        self._adaptive_exact_weights: Dict[int, torch.Tensor] = {}
        self._adaptive_exact_choices: Dict[int, str] = {}
        self._active_token_index = 0
        env_residual_terms = os.environ.get("THINTENSOR_FP8_RESIDUAL_TERMS")
        self._fp8_sparse_residual_terms = max(
            0,
            int(env_residual_terms)
            if env_residual_terms is not None
            else int(fp8_residual_terms),
        )
        if self._adaptive_body_int8 and self.has_fused_gate_up_projection:
            # One exact sparse correction per output row is enough to recover
            # Phi-style fused-projection top-5 ordering in the quick gate while
            # adding only a few MiB, not another full BF16 matrix.
            self._fp8_sparse_residual_terms = max(
                1,
                self._fp8_sparse_residual_terms,
            )
        self._gate_up_sparse_residual_terms = max(
            0,
            int(os.environ.get("THINTENSOR_GATE_UP_RESIDUAL_TERMS", "0")),
        )
        self._int4_sparse_residual_terms = max(
            0, int(os.environ.get("THINTENSOR_INT4_RESIDUAL_TERMS", "0"))
        )
        self._int4_attention_residual_terms = max(
            0,
            int(
                os.environ.get(
                    "THINTENSOR_INT4_ATTENTION_RESIDUAL_TERMS",
                    str(self._int4_sparse_residual_terms),
                )
            ),
        )
        self._int4_lowrank_rank = max(
            0, int(os.environ.get("THINTENSOR_INT4_LOWRANK_RANK", "0"))
        )
        self._int4_lowrank_sidecars: Dict[
            int, tuple[torch.Tensor, torch.Tensor]
        ] = {}
        residual_dtype_name = os.environ.get(
            "THINTENSOR_FP8_RESIDUAL_DTYPE", fp8_residual_dtype
        )
        if residual_dtype_name not in {"bf16", "fp16"}:
            raise ValueError(
                "THINTENSOR_FP8_RESIDUAL_DTYPE must be bf16 or fp16"
            )
        self._fp8_sparse_residual_dtype = (
            torch.float16
            if residual_dtype_name == "fp16"
            else torch.bfloat16
        )
        self._int4_sparse_residual_dtype = (
            torch.float8_e4m3fn
            if os.environ.get("THINTENSOR_INT4_RESIDUAL_DTYPE", "bf16") == "fp8"
            else self._fp8_sparse_residual_dtype
        )
        self._fp8_sparse_residual_ranking = os.environ.get(
            "THINTENSOR_FP8_RESIDUAL_RANKING", fp8_residual_ranking
        )
        if self._fp8_sparse_residual_ranking not in {"raw", "mlp_scale"}:
            raise ValueError(
                "THINTENSOR_FP8_RESIDUAL_RANKING must be raw or mlp_scale"
            )
        self._fp8_sparse_residual_layers = _parse_layer_selection(
            os.environ.get(
                "THINTENSOR_FP8_RESIDUAL_LAYERS",
                fp8_residual_layers or "all",
            ),
            self.layers,
        )
        self._sparse_residual_sidecars: Dict[
            int, tuple[torch.Tensor, torch.Tensor]
        ] = {}
        self.body_fp8_original_bytes = 0
        self.body_fp8_resident_bytes = 0
        if isinstance(self.weights, ThinGpuWeights):
            if not hasattr(self.weights, "_weight_scales"):
                self.weights._weight_scales = {}
            for layer in range(self.layers):
                if self.down_proj_fp8 and layer in self.down_fp8_layers:
                    page_id = _layer_tensor(layer, "mlp.down_proj.weight")
                    q_w, q_s = self._quantize_weight_page(
                        page_id, residual_layer=layer
                    )
                    self._weight_scales[id(q_w)] = q_s
                if self.gate_up_fp8 and layer in self.fp8_layers:
                    gate_up_suffixes = (
                        ("mlp.gate_up_proj.weight",)
                        if self.has_fused_gate_up_projection
                        else ("mlp.gate_proj.weight", "mlp.up_proj.weight")
                    )
                    for suffix in gate_up_suffixes:
                        page_id = _layer_tensor(layer, suffix)
                        q_w, q_s = self._quantize_weight_page(
                            page_id,
                            residual_layer=layer,
                        )
                        self._weight_scales[id(q_w)] = q_s
                if self.qkv_fp8 and layer in self.qkv_fp8_layers:
                    if self.is_qwen3_5:
                        qkv_suffixes = (
                            (
                                "linear_attn.in_proj_qkv.weight",
                                "linear_attn.in_proj_z.weight",
                            )
                            if self.descriptor.layer_types[layer]
                            == "linear_attention"
                            else (
                                "self_attn.q_proj.weight",
                                "self_attn.k_proj.weight",
                                "self_attn.v_proj.weight",
                            )
                        )
                    else:
                        qkv_suffixes = (
                            ("self_attn.qkv_proj.weight",)
                            if self.has_fused_qkv_projection
                            else (
                                "self_attn.q_proj.weight",
                                "self_attn.k_proj.weight",
                                "self_attn.v_proj.weight",
                            )
                        )
                    for suffix in qkv_suffixes:
                        page_id = _layer_tensor(layer, suffix)
                        q_w, q_s = self._quantize_weight_page(page_id)
                        self._weight_scales[id(q_w)] = q_s
                if self.o_proj_fp8 and layer in self.o_fp8_layers:
                    page_id = _layer_tensor(
                        layer,
                        (
                            "linear_attn.out_proj.weight"
                            if self.is_qwen3_5
                            and self.descriptor.layer_types[layer]
                            == "linear_attention"
                            else "self_attn.o_proj.weight"
                        ),
                    )
                    q_w, q_s = self._quantize_weight_page(page_id)
                    self._weight_scales[id(q_w)] = q_s
            for weight_id, scale in self._weight_scales.items():
                quantized = next(
                    (
                        tensor
                        for tensor in self.weights.tensors.values()
                        if id(tensor) == weight_id
                    ),
                    None,
                )
                if quantized is not None:
                    int4_meta = self._int4_metadata.get(weight_id)
                    if int4_meta is not None:
                        rows, cols, _ = int4_meta
                        self.body_fp8_original_bytes += rows * cols * 2
                    elif weight_id in self._mxfp4_metadata:
                        rows, cols, _ = self._mxfp4_metadata[weight_id]
                        self.body_fp8_original_bytes += rows * cols * 2
                    else:
                        self.body_fp8_original_bytes += quantized.numel() * 2
                    self.body_fp8_resident_bytes += (
                        _tensor_nbytes(quantized) + _tensor_nbytes(scale)
                    )
            self.body_fp8_resident_bytes += sum(
                _tensor_nbytes(values) + _tensor_nbytes(indices)
                for values, indices in self._sparse_residual_sidecars.values()
            )
            self.body_fp8_resident_bytes += sum(
                _tensor_nbytes(signs) + _tensor_nbytes(scales)
                for signs, scales in self._mxfp4_binary_residuals.values()
            )
            self.body_fp8_resident_bytes += sum(
                _tensor_nbytes(weight)
                + _tensor_nbytes(scales)
                + _tensor_nbytes(indices)
                for weight, scales, indices
                in self._mxfp4_int8_row_overrides.values()
            )
            self.body_fp8_resident_bytes += sum(
                _tensor_nbytes(scale)
                for scale in self._mxfp4_row_postscales.values()
            )
        elif isinstance(self.weights, ThinGpuPagePool):
            for quantized, scale in self.weights._fp8_cpu_cache.values():
                self.body_fp8_original_bytes += quantized.numel() * 2
                self.body_fp8_resident_bytes += (
                    _tensor_nbytes(quantized) + _tensor_nbytes(scale)
                )
        self.adaptive_exact_resident_bytes = sum(
            _tensor_nbytes(weight)
            for weight in self._adaptive_exact_weights.values()
        )
        if getattr(self, "_use_tensor_cache", False):
            self.weights._use_tensor_cache = True
        self.body_fp8_memory_saved_bytes = max(
            0,
            self.body_fp8_original_bytes
            - self.body_fp8_resident_bytes
            - self.adaptive_exact_resident_bytes,
        )
        self.fp8_sparse_residual_bytes = sum(
            _tensor_nbytes(values) + _tensor_nbytes(indices)
            for values, indices in self._sparse_residual_sidecars.values()
        )
        self.mxfp4_binary_residual_bytes = sum(
            _tensor_nbytes(signs) + _tensor_nbytes(scales)
            for signs, scales in self._mxfp4_binary_residuals.values()
        )
        self.mxfp4_int8_row_override_bytes = sum(
            _tensor_nbytes(weight)
            + _tensor_nbytes(scales)
            + _tensor_nbytes(indices)
            for weight, scales, indices
            in self._mxfp4_int8_row_overrides.values()
        )
        self.mxfp4_row_postscale_bytes = sum(
            _tensor_nbytes(scale)
            for scale in self._mxfp4_row_postscales.values()
        )
        self._layer_plan: Optional[list[LayerPlan]] = None
        if (
            isinstance(self.weights, ThinGpuWeights)
            and not self.is_moe
            and not self.is_qwen3_5
            and not self.is_gemma4
            and not self.has_fused_qkv_projection
            and not self.has_fused_gate_up_projection
        ):
            self._layer_plan = [self._make_layer_plan(layer) for layer in range(self.layers)]
            self._initialize_runtime_fusion()
        elif (
            isinstance(self.weights, ThinGpuPagePool)
            and not self.is_qwen3_5
            and not self.is_gemma4
        ):
            self._layer_plan = [LayerPlan(layer, pool=self.weights) for layer in range(self.layers)]
        if (
            self.device.type == "cuda"
            and self.body_fp8_original_bytes > 0
        ):
            # In-place quantization releases multi-gigabyte BF16 pages, but
            # PyTorch's allocator otherwise keeps those blocks reserved. Give
            # the memory back to CUDA before Triton requests driver scratch.
            torch.cuda.empty_cache()
        if (
            self.kernel_backend is not None
            and self.use_triton_matvec
            and not self.is_moe
            and not self.is_qwen3_5
            and not self.is_gemma4
        ):
            self._autotuning = True
            try:
                self._initialize_matvec_backend_choices()
            finally:
                self._autotuning = False
            if self.lm_head_backend_override:
                self._matvec_choice_by_tensor_id[
                    id(self._lm_head())
                ] = self.lm_head_backend_override
            self._apply_role_backend_overrides()

    def _apply_role_backend_overrides(self) -> None:
        if self._layer_plan is None:
            return
        if isinstance(self.weights, ThinGpuPagePool):
            # Streaming dispatch resolves overrides from stable page ids in
            # _runtime_matvec; touching every lazy property here would
            # materialize the entire model and defeat the residency budget.
            return
        for plan in self._layer_plan:
            if self.gate_up_backend_override:
                self._matvec_choice_by_tensor_id[
                    id(plan.gate_proj)
                ] = self.gate_up_backend_override
                self._matvec_choice_by_tensor_id[
                    id(plan.up_proj)
                ] = self.gate_up_backend_override
            if self.down_proj_backend_override:
                self._matvec_choice_by_tensor_id[
                    id(plan.down_proj)
                ] = self.down_proj_backend_override
            if self.attn_proj_backend_override:
                for weight in (
                    plan.q_proj,
                    plan.k_proj,
                    plan.v_proj,
                    plan.o_proj,
                ):
                    self._matvec_choice_by_tensor_id[
                        id(weight)
                    ] = self.attn_proj_backend_override

    def _quantize_weight_page(
        self,
        page_id: str,
        residual_layer: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source = self.weights.tensors[page_id]
        int4_cache = getattr(self.weights, "_int4_metadata_by_page", {})
        if self._body_int4 and page_id in int4_cache:
            q_s = self.weights._weight_scales.get(page_id)
            if q_s is None:
                raise RuntimeError(
                    f"INT4 scales for {page_id} were not retained"
                )
            rows, cols, group_size = int4_cache[page_id]
            self._int4_metadata[id(source)] = (
                rows,
                cols,
                group_size,
            )
            return source, q_s
        mxfp4_cache = getattr(
            self.weights, "_mxfp4_metadata_by_page", {}
        )
        if page_id in mxfp4_cache:
            q_s = self.weights._weight_scales.get(page_id)
            if q_s is None:
                raise RuntimeError(
                    f"MXFP4 scales for {page_id} were not retained"
                )
            metadata = mxfp4_cache[page_id]
            rows, cols = metadata[:2]
            hadamard_size = int(metadata[2]) if len(metadata) > 2 else 0
            self._mxfp4_metadata[id(source)] = (
                rows,
                cols,
                hadamard_size,
            )
            signs_cache = getattr(
                self.weights, "_mxfp4_hadamard_signs_by_page", {}
            )
            signs = signs_cache.get(page_id)
            if signs is not None:
                self._mxfp4_hadamard_signs[id(source)] = signs
            if (
                self.mxfp4_residual_terms > 0
                or self.mxfp4_binary_residual
                or self.mxfp4_int8_row_fraction > 0
                or self.mxfp4_row_postscale
            ):
                exact_cpu, _ = load_tensor_view(self.weights.archive, page_id)
                needs_sparse = self.mxfp4_residual_terms > 0
                needs_postscale = self.mxfp4_row_postscale
                if not (
                    self.mxfp4_binary_residual
                    or self.mxfp4_int8_row_fraction > 0
                ):
                    sparse, postscale = self._build_mxfp4_sidecars_from_cpu(
                        exact_cpu,
                        source,
                        q_s,
                        hadamard_size=hadamard_size,
                        signs=signs,
                        needs_sparse=needs_sparse,
                        needs_postscale=needs_postscale,
                    )
                    if sparse is not None:
                        self._sparse_residual_sidecars[id(source)] = sparse
                    if postscale is not None:
                        self._mxfp4_row_postscales[id(source)] = postscale
                    return source, q_s
                exact = exact_cpu.to(device=self.device, dtype=self.dtype)
                if needs_sparse:
                    self._sparse_residual_sidecars[id(source)] = self._build_mxfp4_residual(
                        exact,
                        source,
                        q_s,
                        hadamard_size=hadamard_size,
                        signs=signs,
                        terms=self.mxfp4_residual_terms,
                        dtype=self._fp8_sparse_residual_dtype,
                    )
                if self.mxfp4_binary_residual:
                    self._mxfp4_binary_residuals[id(source)] = (
                        self._build_mxfp4_binary_residual(
                            exact,
                            source,
                            q_s,
                            hadamard_size=hadamard_size,
                            signs=signs,
                        )
                    )
                if self.mxfp4_int8_row_fraction > 0:
                    self._mxfp4_int8_row_overrides[id(source)] = (
                        self._build_mxfp4_int8_row_overrides(
                            exact,
                            source,
                            q_s,
                            fraction=self.mxfp4_int8_row_fraction,
                            hadamard_size=hadamard_size,
                            signs=signs,
                        )
                    )
                if needs_postscale:
                    self._mxfp4_row_postscales[id(source)] = (
                        self._build_mxfp4_row_postscale(
                            exact,
                            source,
                            q_s,
                            hadamard_size=hadamard_size,
                            signs=signs,
                        )
                    )
                del exact
            return source, q_s
        page_layer = self.weights.page_specs[page_id].get("layer")
        layer = int(page_layer) if page_layer is not None else None
        mxfp4_selected = layer is not None and (
            (
                any(
                    suffix in page_id
                    for suffix in (
                        "mlp.gate_proj.weight",
                        "mlp.up_proj.weight",
                    )
                )
                and layer in self.mxfp4_gate_up_layers
            )
            or (
                page_id.endswith("mlp.down_proj.weight")
                and layer in self.mxfp4_down_layers
            )
            or (
                any(
                    suffix in page_id
                    for suffix in (
                        "self_attn.q_proj.weight",
                        "self_attn.k_proj.weight",
                        "self_attn.v_proj.weight",
                    )
                )
                and layer in self.mxfp4_qkv_layers
            )
            or (
                page_id.endswith("self_attn.o_proj.weight")
                and layer in self.mxfp4_o_layers
            )
        )
        adaptive_extra = (
            self._adaptive_body_int8
            and (
                (
                    self.is_qwen3_5
                    and (
                        ".linear_attn.in_proj_" in page_id
                        or page_id.endswith(
                            ".linear_attn.out_proj.weight"
                        )
                        or ".self_attn." in page_id
                        or ".mlp." in page_id
                    )
                )
                or
                "self_attn.q_proj.weight" in page_id
                or "self_attn.k_proj.weight" in page_id
                or "self_attn.v_proj.weight" in page_id
                or "self_attn.qkv_proj.weight" in page_id
                or "self_attn.o_proj.weight" in page_id
                or (
                    page_id.endswith("mlp.down_proj.weight")
                    and residual_layer is not None
                    and not 8 <= residual_layer < 28
                )
            )
        )
        fused_gate_up_int8 = (
            self._adaptive_body_int8
            and page_id.endswith("mlp.gate_up_proj.weight")
        )
        if source.dtype in {torch.float8_e4m3fn, torch.int8}:
            q_s = self.weights._weight_scales.get(page_id)
            if q_s is None:
                raise RuntimeError(
                    f"Weight {page_id} is quantized but its scales "
                    "were not found."
                )
            if adaptive_extra:
                exact_cache = getattr(
                    self.weights, "_adaptive_exact_weights", {}
                )
                exact = exact_cache.get(page_id)
                if exact is None:
                    raise RuntimeError(
                        f"Adaptive exact weight for {page_id} was not retained"
                    )
                self._adaptive_exact_weights[id(source)] = exact
                self._adaptive_exact_choices[id(source)] = (
                    "triton_loop_256"
                    if page_id.endswith("mlp.down_proj.weight")
                    else "triton"
                )
            return source, q_s
        scale_block_size = self.fp8_scale_block
        if mxfp4_selected:
            signs = None
            if (
                self.mxfp4_hadamard_size > 0
                and self.mxfp4_hadamard_seed != 0
            ):
                digest = hashlib.blake2b(
                    page_id.encode("utf-8"),
                    digest_size=8,
                ).digest()
                page_seed = int.from_bytes(digest, "little") ^ (
                    self.mxfp4_hadamard_seed & ((1 << 63) - 1)
                )
                generator = torch.Generator(device=source.device)
                generator.manual_seed(page_seed)
                signs = (
                    torch.randint(
                        0,
                        2,
                        (int(source.shape[1]),),
                        device=source.device,
                        generator=generator,
                        dtype=torch.int8,
                    )
                    .mul_(2)
                    .sub_(1)
                    .to(source.dtype)
                )
            q_w, q_s = self._quantize_mxfp4_rows(
                source,
                hadamard_size=self.mxfp4_hadamard_size,
                signs=signs,
                binary_residual=self.mxfp4_binary_residual,
            )
            rows, cols = int(source.shape[0]), int(source.shape[1])
            self._mxfp4_metadata[id(q_w)] = (
                rows,
                cols,
                self.mxfp4_hadamard_size,
            )
            if not hasattr(self.weights, "_mxfp4_metadata_by_page"):
                self.weights._mxfp4_metadata_by_page = {}
            self.weights._mxfp4_metadata_by_page[page_id] = (
                rows,
                cols,
                self.mxfp4_hadamard_size,
            )
            if signs is not None:
                self._mxfp4_hadamard_signs[id(q_w)] = signs
                if not hasattr(
                    self.weights, "_mxfp4_hadamard_signs_by_page"
                ):
                    self.weights._mxfp4_hadamard_signs_by_page = {}
                self.weights._mxfp4_hadamard_signs_by_page[page_id] = signs
            if self.mxfp4_residual_terms > 0:
                self._sparse_residual_sidecars[id(q_w)] = (
                    self._build_mxfp4_residual(
                        source,
                        q_w,
                        q_s,
                        hadamard_size=self.mxfp4_hadamard_size,
                        signs=signs,
                        terms=self.mxfp4_residual_terms,
                        dtype=self._fp8_sparse_residual_dtype,
                    )
                )
            if self.mxfp4_binary_residual:
                self._mxfp4_binary_residuals[id(q_w)] = (
                    self._build_mxfp4_binary_residual(
                        source,
                        q_w,
                        q_s,
                        hadamard_size=self.mxfp4_hadamard_size,
                        signs=signs,
                    )
                )
            if self.mxfp4_int8_row_fraction > 0:
                self._mxfp4_int8_row_overrides[id(q_w)] = (
                    self._build_mxfp4_int8_row_overrides(
                        source,
                        q_w,
                        q_s,
                        fraction=self.mxfp4_int8_row_fraction,
                        hadamard_size=self.mxfp4_hadamard_size,
                        signs=signs,
                    )
                )
            if self.mxfp4_row_postscale:
                self._mxfp4_row_postscales[id(q_w)] = (
                    self._build_mxfp4_row_postscale(
                        source,
                        q_w,
                        q_s,
                        hadamard_size=self.mxfp4_hadamard_size,
                        signs=signs,
                    )
                )
        elif self._body_int4:
            q_w, q_s = self._quantize_int4_rows(
                source,
                group_size=self.body_int4_group_size,
            )
            rows, cols = int(source.shape[0]), int(source.shape[1])
            self._int4_metadata[id(q_w)] = (
                rows,
                cols,
                self.body_int4_group_size,
            )
            if not hasattr(self.weights, "_int4_metadata_by_page"):
                self.weights._int4_metadata_by_page = {}
            self.weights._int4_metadata_by_page[page_id] = (
                rows,
                cols,
                self.body_int4_group_size,
            )
        elif (
            (adaptive_extra or fused_gate_up_int8)
            and self._adaptive_body_int8
        ):
            q_w, q_s = self._quantize_int8_rows(
                source,
                scale_block_size=scale_block_size,
            )
        else:
            q_w, q_s = self._quantize_fp8_rows(
                source,
                scale_block_size=scale_block_size,
            )
        if adaptive_extra:
            if not hasattr(self.weights, "_adaptive_exact_weights"):
                self.weights._adaptive_exact_weights = {}
            self.weights._adaptive_exact_weights[page_id] = source
            self._adaptive_exact_weights[id(q_w)] = source
            self._adaptive_exact_choices[id(q_w)] = (
                "triton_loop_256"
                if page_id.endswith("mlp.down_proj.weight")
                else "triton"
            )
        if (
            (
                page_id.endswith("mlp.down_proj.weight")
                or page_id.endswith("mlp.gate_up_proj.weight")
                or page_id.endswith("mlp.gate_proj.weight")
                or page_id.endswith("mlp.up_proj.weight")
            )
            and q_s.ndim == 1
            and residual_layer in self._fp8_sparse_residual_layers
        ):
            is_gate_up = page_id.endswith(
                (
                    "mlp.gate_up_proj.weight",
                    "mlp.gate_proj.weight",
                    "mlp.up_proj.weight",
                )
            )
            residual_terms = max(
                self._fp8_sparse_residual_terms,
                self._gate_up_sparse_residual_terms if is_gate_up else 0,
                (
                    1
                    if page_id.endswith("mlp.gate_up_proj.weight")
                    and self._adaptive_body_int8
                    else 0
                ),
            )
            if residual_terms > 0:
                terms = min(
                    residual_terms, int(source.shape[1])
                )
                rows = int(source.shape[0])
                residual_values = torch.empty(
                    (rows, terms),
                    device=source.device,
                    dtype=self._fp8_sparse_residual_dtype,
                )
                residual_indices = torch.empty(
                    (rows, terms),
                    device=source.device,
                    dtype=torch.int16,
                )
                channel_importance = None
                if (
                    page_id.endswith("mlp.down_proj.weight")
                    and self._fp8_sparse_residual_ranking == "mlp_scale"
                ):
                    gate_source = self.weights.tensors[
                        page_id.replace("down_proj", "gate_proj")
                    ]
                    up_source = self.weights.tensors[
                        page_id.replace("down_proj", "up_proj")
                    ]
                    channel_importance = (
                        gate_source.float().abs().amax(dim=1)
                        * up_source.float().abs().amax(dim=1)
                    ).sqrt_()
                # Chunking is required when a BF16 model nearly fills VRAM;
                # materializing a full float32 reconstruction would consume
                # hundreds of MiB for a fused gate/up page.
                for start in range(0, rows, 32):
                    end = min(rows, start + 32)
                    error = source[start:end].float().sub_(
                        q_w[start:end].float().mul_(
                            q_s[start:end, None]
                        )
                    )
                    ranking_error = error.abs()
                    if channel_importance is not None:
                        ranking_error.mul_(channel_importance[None, :])
                    _, indices_i64 = torch.topk(
                        ranking_error,
                        k=terms,
                        dim=1,
                        sorted=False,
                    )
                    residual_values[start:end].copy_(
                        torch.gather(error, 1, indices_i64)
                    )
                    residual_indices[start:end].copy_(
                        indices_i64.to(torch.int16)
                    )
                self._sparse_residual_sidecars[id(q_w)] = (
                    residual_values.contiguous(),
                    residual_indices.contiguous(),
                )
        self.weights.tensors[page_id] = q_w
        self.weights._weight_scales[page_id] = q_s
        return q_w, q_s

    def _configure_fp8_head_residency(self, source_head: torch.Tensor) -> None:
        assert self._lm_head_tensor is not None
        embed_is_head = source_head is self._embed_weight or (
            source_head.data_ptr() == self._embed_weight.data_ptr()
        )
        if embed_is_head:
            self.lm_head_tied_to_embeddings = True
            self.lm_head_fp8_separate_execution_head = True
        if self.keep_bf16_lm_head:
            if source_head.element_size() != 2:
                raise RuntimeError(
                    "BF16 lm_head was already replaced; construct the exact-topk "
                    "runtime before replacement or reload the weights"
                )
            self._bf16_lm_head_tensor = source_head
            self.lm_head_bf16_resident = True
            self.lm_head_bf16_bytes = _tensor_nbytes(source_head)
            return

        if embed_is_head:
            # Tied embeddings remain BF16. Quantizing this shared tensor would
            # silently change input embedding numerics, so Head8 uses a
            # separate FP8 execution head and reports the net extra memory.
            return

        if isinstance(self.weights, ThinGpuPagePool):
            if "lm_head.weight" in self.weights.tensors:
                self.weights.drop_persistent_page("lm_head.weight")
        else:
            self.weights.tensors.pop("lm_head.weight", None)

    @staticmethod
    def _quantize_fp8_rows(
        source: torch.Tensor,
        chunk_rows: int = 256,
        scale_block_size: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if source.dtype == torch.float8_e4m3fn:
            raise RuntimeError(
                "FP8 head is already resident but its row scales are unavailable"
            )
        rows = source.shape[0]
        quantized = torch.empty_like(source, dtype=torch.float8_e4m3fn)
        cols = int(source.shape[1])
        scale_blocks = (
            (cols + scale_block_size - 1) // scale_block_size
            if scale_block_size > 0
            else 1
        )
        scales = torch.empty(
            (rows, scale_blocks) if scale_block_size > 0 else (rows,),
            device=source.device,
            dtype=torch.float32,
        )
        fp8_max = torch.finfo(torch.float8_e4m3fn).max
        for start in range(0, rows, chunk_rows):
            end = min(rows, start + chunk_rows)
            source_chunk = source[start:end]
            if scale_block_size > 0:
                for block in range(scale_blocks):
                    col_start = block * scale_block_size
                    col_end = min(cols, col_start + scale_block_size)
                    block_values = source_chunk[:, col_start:col_end]
                    scale_chunk = (
                        block_values.abs()
                        .amax(dim=1)
                        .float()
                        .clamp_min_(1e-12)
                        .div_(fp8_max)
                    )
                    scales[start:end, block].copy_(scale_chunk)
                    quantized[start:end, col_start:col_end].copy_(
                        block_values.float().div_(scale_chunk[:, None])
                    )
            else:
                scale_chunk = (
                    source_chunk.abs()
                    .amax(dim=1)
                    .float()
                    .clamp_min_(1e-12)
                    .div_(fp8_max)
                )
                scales[start:end].copy_(scale_chunk)
                quantized[start:end].copy_(
                    source_chunk.float().div_(scale_chunk[:, None])
                )
        return quantized, scales

    @staticmethod
    def _quantize_int4_rows(
        source: torch.Tensor,
        *,
        group_size: int,
        chunk_rows: int = 256,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rows = int(source.shape[0])
        cols = int(source.shape[1])
        if cols % 2:
            raise ValueError("body INT4 requires an even input width")
        groups = (cols + group_size - 1) // group_size
        packed = torch.empty(
            (rows, cols // 2),
            device=source.device,
            dtype=torch.uint8,
        )
        affine = (
            os.environ.get("THINTENSOR_AFFINE_INT4", "0") == "1"
            and rows < 65536
        )
        scales = torch.empty(
            (rows, groups, 2) if affine else (rows, groups),
            device=source.device,
            dtype=torch.float32,
        )
        for start in range(0, rows, chunk_rows):
            end = min(rows, start + chunk_rows)
            source_chunk = source[start:end].float()
            quantized = torch.empty(
                (end - start, cols),
                device=source.device,
                dtype=torch.uint8,
            )
            for group in range(groups):
                col_start = group * group_size
                col_end = min(cols, col_start + group_size)
                values = source_chunk[:, col_start:col_end]
                if affine:
                    minimum = values.amin(dim=1)
                    scale = (
                        values.amax(dim=1).sub(minimum).clamp_min_(1e-12).div_(15.0)
                    )
                    scales[start:end, group, 0].copy_(scale)
                    scales[start:end, group, 1].copy_(minimum)
                    quantized[:, col_start:col_end].copy_(
                        torch.round((values - minimum[:, None]) / scale[:, None])
                        .clamp_(0, 15)
                    )
                    continue
                scale = (
                    values.abs()
                    .amax(dim=1)
                    .clamp_min_(1e-12)
                    .div_(7.0)
                )
                scales[start:end, group].copy_(scale)
                quantized[:, col_start:col_end].copy_(
                    torch.round(values / scale[:, None])
                    .clamp_(-7, 7)
                    .add_(8)
                )
            packed[start:end].copy_(
                quantized[:, 0::2]
                | (quantized[:, 1::2] << 4)
            )
        return packed, scales

    @staticmethod
    def _quantize_mxfp4_rows(
        source: torch.Tensor,
        *,
        chunk_rows: int = 256,
        hadamard: bool = False,
        hadamard_size: int = 0,
        signs: torch.Tensor | None = None,
        binary_residual: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rows = int(source.shape[0])
        cols = int(source.shape[1])
        if cols % 32:
            raise ValueError("native MXFP4 requires a K dimension divisible by 32")
        groups = cols // 32
        packed = torch.empty(
            (rows, cols // 2),
            device=source.device,
            dtype=torch.uint8,
        )
        scales = torch.empty(
            (rows, groups),
            device=source.device,
            dtype=torch.uint8,
        )
        transform_size = int(hadamard_size) or (32 if hadamard else 0)
        if transform_size and cols % transform_size:
            raise ValueError(
                f"Hadamard block size {transform_size} does not divide {cols}"
            )
        for start in range(0, rows, chunk_rows):
            end = min(rows, start + chunk_rows)
            blocks = source[start:end].float().reshape(
                end - start, groups, 32
            )
            if transform_size:
                transformed = source[start:end].float()
                if signs is not None:
                    transformed = transformed * signs.float()[None, :]
                transformed = transformed.reshape(-1, transform_size)
                stride = 1
                while stride < transform_size:
                    paired = transformed.reshape(
                        -1, transform_size // (2 * stride), 2, stride
                    )
                    left = paired[:, :, 0, :]
                    right = paired[:, :, 1, :]
                    transformed = torch.stack(
                        (left + right, left - right),
                        dim=2,
                    ).reshape(-1, transform_size)
                    stride *= 2
                blocks = transformed.mul_(transform_size**-0.5).reshape(
                    end - start, groups, 32
                )
            base_exponent = torch.ceil(
                torch.log2(
                    blocks.abs().amax(dim=2).clamp_min_(2.0**-126)
                    / 6.0
                )
            ).clamp_(-127, 127)
            codebook = torch.tensor(
                (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0),
                device=source.device,
                dtype=torch.float32,
            )
            best_error = None
            exponent = None
            code = None
            for bias in (-1, 0):
                candidate_exponent = (base_exponent + bias).clamp(
                    -127, 127
                )
                normalized = (
                    blocks
                    / torch.exp2(candidate_exponent)[:, :, None]
                )
                magnitude = normalized.abs()
                candidate_code = (
                    (magnitude >= 0.25).to(torch.uint8)
                    + (magnitude >= 0.75).to(torch.uint8)
                    + (magnitude >= 1.25).to(torch.uint8)
                    + (magnitude >= 1.75).to(torch.uint8)
                    + (magnitude >= 2.5).to(torch.uint8)
                    + (magnitude >= 3.5).to(torch.uint8)
                    + (magnitude >= 5.0).to(torch.uint8)
                )
                candidate_code |= (
                    torch.signbit(normalized).to(torch.uint8) << 3
                )
                reconstructed = (
                    codebook[(candidate_code & 7).long()]
                    * torch.where(
                        (candidate_code & 8) != 0, -1.0, 1.0
                    )
                    * torch.exp2(candidate_exponent)[:, :, None]
                )
                residual = blocks.sub(reconstructed)
                if binary_residual:
                    residual_exponent = torch.round(
                        torch.log2(
                            residual.abs().mean(
                                dim=2, keepdim=True
                            ).clamp_min_(2.0**-126)
                        )
                    ).clamp_(-127, 127)
                    residual_scale = torch.exp2(residual_exponent)
                    residual = residual - torch.where(
                        residual >= 0,
                        residual_scale,
                        -residual_scale,
                    )
                candidate_error = residual.square().sum(dim=2)
                if best_error is None:
                    best_error = candidate_error
                    exponent = candidate_exponent
                    code = candidate_code
                else:
                    better = candidate_error < best_error
                    best_error = torch.where(
                        better, candidate_error, best_error
                    )
                    exponent = torch.where(
                        better, candidate_exponent, exponent
                    )
                    code = torch.where(
                        better[:, :, None], candidate_code, code
                    )
            assert exponent is not None and code is not None
            scales[start:end].copy_(
                exponent.add(127).to(torch.uint8)
            )
            code = code.reshape(end - start, cols)
            packed[start:end].copy_(
                code[:, 0::2] | (code[:, 1::2] << 4)
            )
        return packed, scales

    @staticmethod
    def _build_mxfp4_residual(
        source: torch.Tensor,
        packed: torch.Tensor,
        scales: torch.Tensor,
        *,
        terms: int,
        dtype: torch.dtype,
        hadamard: bool = False,
        hadamard_size: int = 0,
        signs: torch.Tensor | None = None,
        chunk_rows: int = 16,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rows, cols = int(source.shape[0]), int(source.shape[1])
        groups = cols // 32
        terms = min(max(0, int(terms)), cols)
        if terms == 0:
            return (
                torch.empty((rows, 0), device=source.device, dtype=dtype),
                torch.empty((rows, 0), device=source.device, dtype=torch.int16),
            )
        transform_size = int(hadamard_size) or (32 if hadamard else 0)
        residual_values = torch.empty(
            (rows, terms), device=source.device, dtype=dtype
        )
        residual_indices = torch.empty(
            (rows, terms), device=source.device, dtype=torch.int16
        )
        for start in range(0, rows, chunk_rows):
            end = min(rows, start + chunk_rows)
            target = source[start:end].float().reshape(
                end - start, groups, 32
            )
            if transform_size:
                transformed = source[start:end].float()
                if signs is not None:
                    transformed = transformed * signs.float()[None, :]
                transformed = transformed.reshape(-1, transform_size)
                stride = 1
                while stride < transform_size:
                    paired = transformed.reshape(
                        -1, transform_size // (2 * stride), 2, stride
                    )
                    left = paired[:, :, 0, :]
                    right = paired[:, :, 1, :]
                    transformed = torch.stack(
                        (left + right, left - right),
                        dim=2,
                    ).reshape(-1, transform_size)
                    stride *= 2
                target = transformed.mul_(transform_size**-0.5).reshape(
                    end - start, groups, 32
                )
            packed_chunk = packed[start:end].reshape(
                end - start, groups, 16
            )
            codebook = torch.tensor(
                (
                    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
                ),
                device=source.device,
                dtype=torch.float32,
            )
            reconstructed = torch.empty(
                (end - start, groups, 32),
                device=source.device,
                dtype=torch.float32,
            )
            reconstructed[..., 0::2] = codebook[
                (packed_chunk & 0x0F).long()
            ]
            reconstructed[..., 1::2] = codebook[
                (packed_chunk >> 4).long()
            ]
            reconstructed = torch.ldexp(
                reconstructed,
                scales[start:end].to(torch.int32).sub(127).unsqueeze(-1),
            )
            error = target.reshape(end - start, cols).sub_(
                reconstructed.reshape(end - start, cols)
            )
            _, selected = torch.topk(
                error.abs(),
                k=terms,
                dim=1,
                sorted=False,
            )
            residual_values[start:end].copy_(
                torch.gather(error, 1, selected).to(dtype=dtype)
            )
            residual_indices[start:end].copy_(selected.to(torch.int16))
        return residual_values.contiguous(), residual_indices.contiguous()

    def _build_mxfp4_sidecars_from_cpu(
        self,
        source_cpu: torch.Tensor,
        packed: torch.Tensor,
        scales: torch.Tensor,
        *,
        hadamard_size: int = 0,
        signs: torch.Tensor | None = None,
        needs_sparse: bool,
        needs_postscale: bool,
        chunk_rows: int = 16,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor] | None, torch.Tensor | None]:
        """Build MXFP4 correction data without staging a whole matrix on VRAM."""
        rows = int(source_cpu.shape[0])
        terms = min(max(0, int(self.mxfp4_residual_terms)), int(source_cpu.shape[1]))
        residual_values = residual_indices = None
        if needs_sparse:
            residual_values = torch.empty(
                (rows, terms), device=self.device, dtype=self._fp8_sparse_residual_dtype
            )
            residual_indices = torch.empty(
                (rows, terms), device=self.device, dtype=torch.int16
            )
        postscale = (
            torch.empty(rows, device=self.device, dtype=torch.bfloat16)
            if needs_postscale
            else None
        )
        for start in range(0, rows, chunk_rows):
            end = min(rows, start + chunk_rows)
            exact = source_cpu[start:end].to(
                device=self.device,
                dtype=self._final_norm_weight.dtype,
            )
            if needs_sparse:
                values, indices = self._build_mxfp4_residual(
                    exact,
                    packed[start:end],
                    scales[start:end],
                    hadamard_size=hadamard_size,
                    signs=signs,
                    terms=terms,
                    dtype=self._fp8_sparse_residual_dtype,
                )
                assert residual_values is not None and residual_indices is not None
                residual_values[start:end].copy_(values)
                residual_indices[start:end].copy_(indices)
                del values, indices
            if needs_postscale:
                assert postscale is not None
                postscale[start:end].copy_(
                    self._build_mxfp4_row_postscale(
                        exact,
                        packed[start:end],
                        scales[start:end],
                        hadamard_size=hadamard_size,
                        signs=signs,
                    )
                )
            del exact
        sparse = (
            (residual_values.contiguous(), residual_indices.contiguous())
            if residual_values is not None and residual_indices is not None
            else None
        )
        # Sidecars are persistent, but the per-chunk reconstruction buffers are
        # not. Return their cached segments before constructing the next matrix
        # so a nearly-full resident model does not fail from allocator
        # fragmentation despite having enough aggregate free memory.
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return sparse, postscale

    def _build_int4_residual_from_cpu(
        self,
        source_cpu: torch.Tensor,
        packed: torch.Tensor,
        scales: torch.Tensor,
        *,
        terms: int,
        chunk_rows: int = 16,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rows, cols = map(int, source_cpu.shape)
        terms = min(max(0, int(terms)), cols)
        values_out = torch.empty(
            (rows, terms), device=self.device, dtype=self._int4_sparse_residual_dtype
        )
        indices_out = torch.empty(
            (rows, terms), device=self.device, dtype=torch.int16
        )
        affine = scales.ndim == 3
        group_size = cols // int(scales.shape[1])
        for start in range(0, rows, chunk_rows):
            end = min(rows, start + chunk_rows)
            exact = source_cpu[start:end].to(device=self.device, dtype=torch.float32)
            packed_chunk = packed[start:end]
            codes = torch.empty(
                (end - start, cols), device=self.device, dtype=torch.float32
            )
            codes[:, 0::2] = (packed_chunk & 0x0F).float()
            codes[:, 1::2] = (packed_chunk >> 4).float()
            grouped = codes.reshape(end - start, -1, group_size)
            if affine:
                scale_chunk = scales[start:end, :, 0].float()
                offset_chunk = scales[start:end, :, 1].float()
                reconstructed = grouped * scale_chunk[:, :, None] + offset_chunk[:, :, None]
            else:
                scale_chunk = scales[start:end].float()
                reconstructed = (grouped - 8.0) * scale_chunk[:, :, None]
            error = exact.sub(reconstructed.reshape(end - start, cols))
            _, selected = torch.topk(error.abs(), k=terms, dim=1, sorted=False)
            values_out[start:end].copy_(
                torch.gather(error, 1, selected).to(self._int4_sparse_residual_dtype)
            )
            indices_out[start:end].copy_(selected.to(torch.int16))
            del exact, codes, grouped, reconstructed, error, selected
        torch.cuda.empty_cache()
        return values_out.contiguous(), indices_out.contiguous()

    def _build_int4_lowrank_from_cpu(
        self,
        source_cpu: torch.Tensor,
        packed: torch.Tensor,
        scales: torch.Tensor,
        *,
        rank: int,
        chunk_rows: int = 32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Randomized low-rank correction for grouped-INT4 matrix error."""
        rows, cols = map(int, source_cpu.shape)
        rank = min(max(1, int(rank)), rows, cols)
        affine = scales.ndim == 3
        group_size = cols // int(scales.shape[1])
        generator = torch.Generator(device=self.device)
        generator.manual_seed(rows * 1315423911 + cols)
        omega = torch.randn(
            (cols, rank), device=self.device, dtype=torch.float32, generator=generator
        )
        y = torch.empty((rows, rank), device=self.device, dtype=torch.float32)

        def error_chunk(start: int, end: int) -> torch.Tensor:
            exact = source_cpu[start:end].to(device=self.device, dtype=torch.float32)
            packed_chunk = packed[start:end]
            codes = torch.empty(
                (end - start, cols), device=self.device, dtype=torch.float32
            )
            codes[:, 0::2] = (packed_chunk & 0x0F).float()
            codes[:, 1::2] = (packed_chunk >> 4).float()
            grouped = codes.reshape(end - start, -1, group_size)
            if affine:
                reconstructed = (
                    grouped * scales[start:end, :, 0].float()[:, :, None]
                    + scales[start:end, :, 1].float()[:, :, None]
                )
            else:
                reconstructed = (
                    (grouped - 8.0) * scales[start:end].float()[:, :, None]
                )
            return exact.sub(reconstructed.reshape(end - start, cols))

        for start in range(0, rows, chunk_rows):
            end = min(rows, start + chunk_rows)
            error = error_chunk(start, end)
            y[start:end].copy_(error @ omega)
            del error
        # One power iteration sharply improves the captured error subspace for
        # the same persistent rank, important when a few correlated weight
        # directions dominate otherwise diffuse grouped-quantization noise.
        z = torch.zeros((cols, rank), device=self.device, dtype=torch.float32)
        for start in range(0, rows, chunk_rows):
            end = min(rows, start + chunk_rows)
            error = error_chunk(start, end)
            z.add_(error.transpose(0, 1) @ y[start:end])
            del error
        for start in range(0, rows, chunk_rows):
            end = min(rows, start + chunk_rows)
            error = error_chunk(start, end)
            y[start:end].copy_(error @ z)
            del error
        del z
        q_cpu, _ = torch.linalg.qr(y.cpu(), mode="reduced")
        q = q_cpu.to(device=self.device)
        del y, omega, q_cpu
        b = torch.zeros((rank, cols), device=self.device, dtype=torch.float32)
        for start in range(0, rows, chunk_rows):
            end = min(rows, start + chunk_rows)
            error = error_chunk(start, end)
            b.add_(q[start:end].transpose(0, 1) @ error)
            del error
        u_small_cpu, singular_cpu, vh_cpu = torch.linalg.svd(
            b.cpu(), full_matrices=False
        )
        u_small = u_small_cpu.to(device=self.device)
        singular = singular_cpu.to(device=self.device)
        u = (q @ u_small).mul_(singular[None, :]).to(torch.bfloat16)
        v = vh_cpu.to(device=self.device, dtype=torch.bfloat16)
        del q, b, u_small, singular, u_small_cpu, singular_cpu, vh_cpu
        torch.cuda.empty_cache()
        return u.contiguous(), v.contiguous()

    @staticmethod
    def _build_mxfp4_binary_residual(
        source: torch.Tensor,
        packed: torch.Tensor,
        scales: torch.Tensor,
        *,
        hadamard_size: int = 0,
        signs: torch.Tensor | None = None,
        chunk_rows: int = 64,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rows, cols = int(source.shape[0]), int(source.shape[1])
        groups = cols // 32
        packed_signs = torch.empty(
            (rows, cols // 8),
            device=source.device,
            dtype=torch.uint8,
        )
        residual_scales = torch.empty(
            (rows, groups),
            device=source.device,
            dtype=torch.uint8,
        )
        codebook = torch.tensor(
            (
                0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
            ),
            device=source.device,
            dtype=torch.float32,
        )
        for start in range(0, rows, chunk_rows):
            end = min(rows, start + chunk_rows)
            target = source[start:end].float()
            if signs is not None:
                target = target * signs.float()[None, :]
            if hadamard_size:
                transformed = target.reshape(-1, hadamard_size)
                stride = 1
                while stride < hadamard_size:
                    paired = transformed.reshape(
                        -1, hadamard_size // (2 * stride), 2, stride
                    )
                    left = paired[:, :, 0, :]
                    right = paired[:, :, 1, :]
                    transformed = torch.stack(
                        (left + right, left - right),
                        dim=2,
                    ).reshape(-1, hadamard_size)
                    stride *= 2
                target = transformed.mul_(hadamard_size**-0.5).reshape(
                    end - start, cols
                )
            packed_chunk = packed[start:end].reshape(
                end - start, groups, 16
            )
            reconstructed = torch.empty(
                (end - start, groups, 32),
                device=source.device,
                dtype=torch.float32,
            )
            reconstructed[..., 0::2] = codebook[
                (packed_chunk & 0x0F).long()
            ]
            reconstructed[..., 1::2] = codebook[
                (packed_chunk >> 4).long()
            ]
            reconstructed = torch.ldexp(
                reconstructed,
                scales[start:end].to(torch.int32).sub(127).unsqueeze(-1),
            ).reshape(end - start, cols)
            error = target.sub(reconstructed)
            group_scale = error.abs().reshape(
                end - start, groups, 32
            ).mean(dim=2)
            group_exponent = torch.round(
                torch.log2(group_scale.clamp_min_(2.0**-126))
            ).clamp_(-127, 127)
            residual_scales[start:end].copy_(
                group_exponent.add(127).to(torch.uint8)
            )
            positive = (error >= 0).to(torch.uint8)
            packed_chunk_signs = positive[:, 0::8]
            for bit in range(1, 8):
                packed_chunk_signs = packed_chunk_signs | (
                    positive[:, bit::8] << bit
                )
            packed_signs[start:end].copy_(packed_chunk_signs)
        return packed_signs, residual_scales

    @staticmethod
    def _build_mxfp4_int8_row_overrides(
        source: torch.Tensor,
        packed: torch.Tensor,
        scales: torch.Tensor,
        *,
        fraction: float,
        hadamard_size: int = 0,
        signs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rows, cols = int(source.shape[0]), int(source.shape[1])
        groups = cols // 32
        codebook = torch.tensor(
            (
                0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
            ),
            device=source.device,
            dtype=torch.float32,
        )
        packed_groups = packed.reshape(rows, groups, 16)
        reconstructed = torch.empty(
            (rows, groups, 32),
            device=source.device,
            dtype=torch.float32,
        )
        reconstructed[..., 0::2] = codebook[
            (packed_groups & 0x0F).long()
        ]
        reconstructed[..., 1::2] = codebook[
            (packed_groups >> 4).long()
        ]
        reconstructed = torch.ldexp(
            reconstructed,
            scales.to(torch.int32).sub(127).unsqueeze(-1),
        ).reshape(rows, cols)
        target = source.float()
        if signs is not None:
            target = target * signs.float()[None, :]
        if hadamard_size:
            transformed = target.reshape(-1, hadamard_size)
            stride = 1
            while stride < hadamard_size:
                paired = transformed.reshape(
                    -1, hadamard_size // (2 * stride), 2, stride
                )
                left = paired[:, :, 0, :]
                right = paired[:, :, 1, :]
                transformed = torch.stack(
                    (left + right, left - right),
                    dim=2,
                ).reshape(-1, hadamard_size)
                stride *= 2
            target = transformed.mul_(hadamard_size**-0.5).reshape(
                rows, cols
            )
        relative_error = target.sub(reconstructed).square().sum(dim=1)
        relative_error.div_(target.square().sum(dim=1).clamp_min_(1e-20))
        selected_rows = min(
            rows,
            max(1, int(round(rows * fraction))),
        )
        row_indices = torch.topk(
            relative_error,
            k=selected_rows,
            sorted=True,
        ).indices
        exact_rows = source.index_select(0, row_indices)
        row_scales = (
            exact_rows.abs()
            .amax(dim=1)
            .float()
            .clamp_min_(1e-12)
            .div_(127.0)
        )
        int8_rows = torch.round(
            exact_rows.float() / row_scales[:, None]
        ).clamp_(-127, 127).to(torch.int8)
        return (
            int8_rows.contiguous(),
            row_scales.contiguous(),
            row_indices.to(torch.int32).contiguous(),
        )

    @staticmethod
    def _build_mxfp4_row_postscale(
        source: torch.Tensor,
        packed: torch.Tensor,
        scales: torch.Tensor,
        *,
        hadamard_size: int = 0,
        signs: torch.Tensor | None = None,
        chunk_rows: int = 64,
    ) -> torch.Tensor:
        rows, cols = int(source.shape[0]), int(source.shape[1])
        groups = cols // 32
        postscale = torch.empty(
            rows,
            device=source.device,
            dtype=torch.bfloat16,
        )
        codebook = torch.tensor(
            (
                0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
            ),
            device=source.device,
            dtype=torch.float32,
        )
        for start in range(0, rows, chunk_rows):
            end = min(rows, start + chunk_rows)
            target = source[start:end].float()
            if signs is not None:
                target = target * signs.float()[None, :]
            if hadamard_size:
                transformed = target.reshape(-1, hadamard_size)
                stride = 1
                while stride < hadamard_size:
                    paired = transformed.reshape(
                        -1, hadamard_size // (2 * stride), 2, stride
                    )
                    left = paired[:, :, 0, :]
                    right = paired[:, :, 1, :]
                    transformed = torch.stack(
                        (left + right, left - right),
                        dim=2,
                    ).reshape(-1, hadamard_size)
                    stride *= 2
                target = transformed.mul_(hadamard_size**-0.5).reshape(
                    end - start, cols
                )
            packed_chunk = packed[start:end].reshape(
                end - start, groups, 16
            )
            reconstructed = torch.empty(
                (end - start, groups, 32),
                device=source.device,
                dtype=torch.float32,
            )
            reconstructed[..., 0::2] = codebook[
                (packed_chunk & 0x0F).long()
            ]
            reconstructed[..., 1::2] = codebook[
                (packed_chunk >> 4).long()
            ]
            reconstructed = torch.ldexp(
                reconstructed,
                scales[start:end].to(torch.int32).sub(127).unsqueeze(-1),
            ).reshape(end - start, cols)
            numerator = (target * reconstructed).sum(dim=1)
            denominator = reconstructed.square().sum(dim=1).clamp_min_(1e-20)
            postscale[start:end].copy_(numerator.div_(denominator))
        return postscale

    def _quantize_int8_rows(
        self,
        source: torch.Tensor,
        *,
        scale_block_size: int = 0,
        chunk_rows: int = 256,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rows = int(source.shape[0])
        cols = int(source.shape[1])
        int8_scale_max = 127.0
        scale_blocks = (
            (cols + scale_block_size - 1) // scale_block_size
            if scale_block_size > 0
            else 1
        )
        quantized = torch.empty_like(source, dtype=torch.int8)
        scales = torch.empty(
            (rows, scale_blocks) if scale_block_size > 0 else (rows,),
            device=source.device,
            dtype=torch.float32,
        )
        for start in range(0, rows, chunk_rows):
            end = min(rows, start + chunk_rows)
            source_chunk = source[start:end]
            if scale_block_size > 0:
                for block in range(scale_blocks):
                    col_start = block * scale_block_size
                    col_end = min(cols, col_start + scale_block_size)
                    block_values = source_chunk[:, col_start:col_end]
                    scale_chunk = (
                        block_values.abs()
                        .amax(dim=1)
                        .float()
                        .clamp_min_(1e-12)
                        .div_(int8_scale_max)
                    )
                    scales[start:end, block].copy_(scale_chunk)
                    quantized[start:end, col_start:col_end].copy_(
                        torch.round(
                            block_values.float() / scale_chunk[:, None]
                        ).clamp_(-127, 127)
                    )
            else:
                scale_chunk = (
                    source_chunk.abs()
                    .amax(dim=1)
                    .float()
                    .clamp_min_(1e-12)
                    .div_(int8_scale_max)
                )
                scales[start:end].copy_(scale_chunk)
                quantized[start:end].copy_(
                    torch.round(
                        source_chunk.float() / scale_chunk[:, None]
                    ).clamp_(-127, 127)
                )
        return quantized, scales

    def _infer_head_dim(self) -> int:
        candidates: list[int] = []
        if self.manifest_head_dim is not None:
            candidates.append(self.manifest_head_dim)
        if (
            self.manifest_kv_heads > 0
            and self.kv_dim % self.manifest_kv_heads == 0
        ):
            candidates.append(self.kv_dim // self.manifest_kv_heads)
        if self.manifest_heads > 0 and self.q_dim % self.manifest_heads == 0:
            candidates.append(self.q_dim // self.manifest_heads)
        candidates.extend([128, 96, 80, 64, 256])

        seen = set()
        for candidate in candidates:
            if candidate in seen or candidate <= 0:
                continue
            seen.add(candidate)
            if self.q_dim % candidate == 0 and self.kv_dim % candidate == 0:
                return candidate
        raise RuntimeError(
            "cannot infer attention head_dim: "
            f"q_dim={self.q_dim}, kv_dim={self.kv_dim}, "
            f"manifest_heads={self.manifest_heads}, "
            f"manifest_kv_heads={self.manifest_kv_heads}, "
            f"manifest_head_dim={self.manifest_head_dim}"
        )

    @property
    def fused_mlp_enabled(self) -> bool:
        return (
            self.fused_mlp_enabled_flag
            or self.fused_scaled_mlp_enabled_flag
        )

    @property
    def fused_mlp_supported_layers(self) -> int:
        return self.layers if self.fused_mlp_enabled_flag else 0

    @property
    def fused_mlp_extra_bytes(self) -> int:
        return 0

    @property
    def fused_mlp_fallbacks(self) -> int:
        return self._fused_mlp_fallbacks

    @property
    def mlp_time_ms(self) -> Optional[float]:
        if getattr(self, "_profiler_enabled", False):
            res = self.get_profiler_results()
            return res.get("mlp_total_time_ms")
        return None

    def get_profiler_results(self) -> dict:
        torch.cuda.synchronize()
        if not self._profile_steps:
            return {}

        step_count = len(self._profile_steps)
        layer_count = 0
        totals: dict[str, float] = {}
        unavailable: dict[str, str] = {}

        def record(start: Any, end: Any, label: str) -> bool:
            try:
                totals[label] = totals.get(label, 0.0) + float(
                    start.elapsed_time(end)
                )
                return True
            except (RuntimeError, ValueError, TypeError) as exc:
                unavailable.setdefault(label, str(exc))
                return False

        for step in self._profile_steps:
            if step["forward_start"] is not None and step["forward_end"] is not None:
                record(step["forward_start"], step["forward_end"], "forward")
            if step["lm_head_start"] is not None and step["lm_head_end"] is not None:
                record(step["lm_head_start"], step["lm_head_end"], "lm_head")

            for lev in step["layers"]:
                layer_count += 1
                record(lev.layer_start, lev.layer_end, "layer")
                record(lev.attn_start, lev.attn_end, "attention")
                record(lev.qkv_start, lev.qkv_end, "qkv")
                record(lev.qk_start, lev.qk_end, "qk")
                record(lev.softmax_start, lev.softmax_end, "softmax")
                record(lev.value_mix_start, lev.value_mix_end, "value_mix")
                record(lev.o_proj_start, lev.o_proj_end, "o_proj")
                record(lev.mlp_start, lev.mlp_end, "mlp")

                # Check fused MLP vs separate events
                is_fused_recorded = False
                if lev.fused_mlp_start is not None and lev.fused_mlp_end is not None:
                    is_fused_recorded = record(
                        lev.fused_mlp_start,
                        lev.fused_mlp_end,
                        "gate_proj",
                    )

                if not is_fused_recorded:
                    if lev.gate_proj_start is not None and lev.gate_proj_end is not None:
                        record(lev.gate_proj_start, lev.gate_proj_end, "gate_proj")
                    if lev.up_proj_start is not None and lev.up_proj_end is not None:
                        record(lev.up_proj_start, lev.up_proj_end, "up_proj")
                    if lev.silu_mul_start is not None and lev.silu_mul_end is not None:
                        record(lev.silu_mul_start, lev.silu_mul_end, "silu_mul")

                if lev.down_proj_start is not None and lev.down_proj_end is not None:
                    record(lev.down_proj_start, lev.down_proj_end, "down_proj")

        denom = layer_count if layer_count > 0 else 1
        def average(label: str, divisor: int) -> float | None:
            if label in unavailable:
                return None
            return totals.get(label, 0.0) / max(1, divisor)

        return {
            "total_forward_token_time_ms": average("forward", step_count),
            "per_layer_average_time_ms": average("layer", denom),
            "attention_time_ms": average("attention", denom),
            "qkv_time_ms": average("qkv", denom),
            "qk_time_ms": average("qk", denom),
            "softmax_time_ms": average("softmax", denom),
            "value_mix_time_ms": average("value_mix", denom),
            "o_proj_time_ms": average("o_proj", denom),
            "mlp_total_time_ms": average("mlp", denom),
            "gate_proj_time_ms": average("gate_proj", denom),
            "up_proj_time_ms": average("up_proj", denom),
            "silu_mul_time_ms": average("silu_mul", denom),
            "down_proj_time_ms": average("down_proj", denom),
            "lm_head_argmax_time_ms": average("lm_head", step_count),
            "timing_unavailable": unavailable,
        }

    @property
    def geometry_telemetry(self) -> dict[str, Any]:
        return {
            "runtime_heads": self.heads,
            "runtime_kv_heads": self.kv_heads,
            "runtime_head_dim": self.head_dim,
            "runtime_q_dim": self.q_dim,
            "runtime_kv_dim": self.kv_dim,
            "manifest_heads": self.manifest_heads,
            "manifest_kv_heads": self.manifest_kv_heads,
            "manifest_head_dim": self.manifest_head_dim,
            "model_type": self.descriptor.model_type,
            "attention_mode": self.attention_mode,
            "attention_backend": self.attention_backend,
            "model_moe_top_k": self.model_moe_top_k,
            "moe_top_k_limit": self.moe_top_k_limit,
            "moe_top_k_renorm": self.moe_top_k_renorm,
            "effective_moe_top_k": self.effective_moe_top_k,
            "kv_tokens_attended": self._last_kv_tokens_attended,
            "kv_cache_read_bytes_per_token": (
                self.kv_cache.read_bytes_per_token
                if self.kv_cache is not None
                else 0
            ),
            "not_hf_equivalent": self.not_hf_equivalent,
            "reason_not_equivalent": self.reason_not_equivalent,
            "execution_schema": self.execution_plan.schema,
            "operator_contract": sorted(self.operator_contract),
            "runtime_executor": self.runtime_executor.name,
        }

    @torch.inference_mode()
    def begin_decode(self, token_index: int) -> None:
        """Anchor adaptive precision to the first generated-token position."""
        if self.attention_mode == "causal_kv" and self.kv_cache is not None:
            self._last_kv_tokens_attended = int(token_index)
        if self._adaptive_fp8_start_token < 0:
            return
        if token_index < 0:
            raise ValueError("decode token index must be non-negative")
        self._adaptive_switch_token_index = (
            int(token_index) + self._adaptive_fp8_start_token
        )

    def _bound_layer_tensor(
        self,
        layer: int,
        role: str,
    ) -> Optional[torch.Tensor]:
        if layer >= len(self._bound_layer_tensors):
            return None
        return self._bound_layer_tensors[layer].get(role)

    @torch.inference_mode()
    def forward_token(
        self,
        token_id: int | torch.Tensor,
        layers: Optional[int] = None,
        token_index: int = 0,
    ) -> torch.Tensor:
        if hasattr(self.weights, "decode_steps_run"):
            self.weights.decode_steps_run += 1
        if getattr(self, "_profiler_enabled", False):
            self._current_step_profile = {
                "layers": [],
                "forward_start": torch.cuda.Event(enable_timing=True),
                "forward_end": torch.cuda.Event(enable_timing=True),
                "lm_head_start": None,
                "lm_head_end": None,
            }
            self._profile_steps.append(self._current_step_profile)
            self._current_step_profile["forward_start"].record()

        embed = self._embed_weight
        if isinstance(token_id, int) and (token_id < 0 or token_id >= embed.shape[0]):
            raise ValueError(f"token_id {token_id} outside vocab {embed.shape[0]}")
        layer_count = min(self.layers, layers if layers is not None else self.layers)
        
        res = self._execute_operator_graph(
            embed,
            token_id,
            layer_count,
            token_index,
        )

        if getattr(self, "_profiler_enabled", False):
            self._current_step_profile["forward_end"].record()

        return res

    def _execute_operator_graph(
        self,
        embed: torch.Tensor,
        token_id: int | torch.Tensor,
        layer_count: int,
        token_index: int,
    ) -> torch.Tensor:
        """Dispatch an executable graph by semantic blocks, not model family."""
        graph = self.runtime_executor
        if graph.name != "operator_graph_decoder":
            raise RuntimeError(
                f"unregistered runtime executor {graph.name!r}"
            )
        layers = graph.layers[:layer_count]
        sequence_blocks = {layer.sequence for layer in layers}
        feedforward_blocks = {layer.feedforward for layer in layers}
        if "shared_kv_attention" in sequence_blocks:
            return self._forward_token_gemma4(
                embed, token_id, layer_count, token_index
            )
        if "gated_delta_net" in sequence_blocks:
            return self._forward_token_qwen3_5(
                embed, token_id, layer_count, token_index
            )
        if "sparse_moe" in feedforward_blocks:
            return self._forward_token_moe(
                embed, token_id, layer_count, token_index
            )
        if self.kernel_backend is not None and all(
            not layer.schema.startswith("fused_") for layer in layers
        ) and not self.uses_dual_post_norm:
            return self._forward_token_triton(
                embed, token_id, layer_count, token_index
            )
        return self._forward_token_pytorch_fallback(
            embed, token_id, layer_count, token_index
        )

    @torch.inference_mode()
    def _forward_token_gemma4(
        self,
        embed: torch.Tensor,
        token_id: int | torch.Tensor,
        layer_count: int,
        token_index: int,
    ) -> torch.Tensor:
        if token_index == 0:
            for values in self._gemma4_key_history.values():
                values.clear()
            for values in self._gemma4_value_history.values():
                values.clear()
        if isinstance(token_id, torch.Tensor):
            token = token_id.reshape(1).to(
                device=embed.device, dtype=torch.long
            )
            hidden = embed.index_select(0, token).reshape(-1).clone()
        else:
            token = torch.tensor(
                [token_id], device=embed.device, dtype=torch.long
            )
            hidden = embed[token_id].clone()
        if getattr(self, "_gemma4_normed_buffer", None) is None:
            self._gemma4_normed_buffer = torch.empty_like(hidden)
            self._gemma4_normed_buffer_2 = torch.empty_like(hidden)
        hidden.mul_(
            torch.tensor(
                self.embedding_scale,
                device=hidden.device,
                dtype=hidden.dtype,
            )
        )
        per_layer_inputs = self._gemma4_per_layer_inputs(
            token, hidden
        )

        for layer in range(layer_count):
            residual = hidden
            normed = self._normalization(
                hidden,
                _layer_tensor(layer, "input_layernorm.weight"),
                out=self._gemma4_normed_buffer,
            )
            attention = self._gemma4_attention(
                layer, normed, token_index
            )
            attention = self._normalization(
                attention,
                _layer_tensor(
                    layer, "post_attention_layernorm.weight"
                ),
                out=self._gemma4_normed_buffer_2,
            )
            hidden = residual + attention

            residual = hidden
            normed = self._normalization(
                hidden,
                _layer_tensor(
                    layer, "pre_feedforward_layernorm.weight"
                ),
                out=self._gemma4_normed_buffer,
            )
            gate_weight = self.weights.tensor(
                _layer_tensor(layer, "mlp.gate_proj.weight")
            )
            up_weight = self.weights.tensor(
                _layer_tensor(layer, "mlp.up_proj.weight")
            )
            intermediate = int(gate_weight.shape[0])
            if self.kernel_backend is not None and self.gemma4_triton_mlp:
                assert self._gemma4_gate_up_buffer is not None
                gate_up = self._gemma4_gate_up_buffer[
                    : 2 * intermediate
                ]
                self.kernel_backend.multi_matvec(
                    (gate_weight, up_weight), normed, gate_up
                )
                gate = gate_up[:intermediate]
                up = gate_up[intermediate:]
            else:
                gate = torch.mv(gate_weight, normed)
                up = torch.mv(up_weight, normed)
            activation = torch.nn.functional.gelu(
                gate, approximate="tanh"
            ) * up
            down_weight = self.weights.tensor(
                _layer_tensor(layer, "mlp.down_proj.weight")
            )
            if self.kernel_backend is not None and self.gemma4_triton_mlp:
                assert self._gemma4_projection_buffer is not None
                mlp = self.kernel_backend.matvec(
                    down_weight,
                    activation,
                    self._gemma4_projection_buffer,
                    config_name="triton_loop_256",
                    block_m=16,
                    num_warps=4,
                )
            else:
                mlp = torch.mv(down_weight, activation)
            mlp = self._normalization(
                mlp,
                _layer_tensor(
                    layer, "post_feedforward_layernorm.weight"
                ),
                out=self._gemma4_normed_buffer_2,
            )
            hidden = residual + mlp

            ple_gate_weight = self.weights.tensor(
                _layer_tensor(
                    layer, "per_layer_input_gate.weight"
                )
            )
            if self._gemma4_ple_gate_buffer is not None:
                ple_gate = torch.mv(ple_gate_weight, hidden, out=self._gemma4_ple_gate_buffer)
            else:
                ple_gate = torch.mv(ple_gate_weight, hidden)
            ple_gate = torch.nn.functional.gelu(
                ple_gate, approximate="tanh"
            )
            ple_gate.mul_(per_layer_inputs[layer])
            ple_projection = self.weights.tensor(
                _layer_tensor(
                    layer, "per_layer_projection.weight"
                )
            )
            if self._gemma4_projection_buffer is not None:
                ple = torch.mv(ple_projection, ple_gate, out=self._gemma4_projection_buffer)
            else:
                ple = torch.mv(ple_projection, ple_gate)
            ple = self._normalization(
                ple,
                _layer_tensor(
                    layer, "post_per_layer_input_norm.weight"
                ),
                out=self._gemma4_normed_buffer_2,
            )
            hidden = hidden + ple
            layer_scalar = self.weights.tensor(
                _layer_tensor(layer, "layer_scalar")
            )
            hidden.mul_(layer_scalar.reshape(()))

        return self._normalization(hidden, "model.norm.weight")

    def _gemma4_per_layer_inputs(
        self,
        token: torch.Tensor,
        inputs_embed: torch.Tensor,
    ) -> torch.Tensor:
        page_id = "model.embed_tokens_per_layer.weight"
        table = self.weights.tensor(page_id)
        scales = (
            self.weights.rowwise_int8_scale(page_id)
            if hasattr(self.weights, "rowwise_int8_scale")
            else None
        )
        if scales is None:
            token_component = table.index_select(0, token).reshape(
                self.layers,
                self.descriptor.hidden_size_per_layer_input,
            )
        else:
            token_component = (
                table.index_select(0, token).reshape(-1).to(
                    dtype=inputs_embed.dtype
                )
                * scales.index_select(0, token).reshape(()).to(
                    dtype=inputs_embed.dtype
                )
            ).reshape(
                self.layers,
                self.descriptor.hidden_size_per_layer_input,
            )
        token_component.mul_(
            torch.tensor(
                self.descriptor.hidden_size_per_layer_input**0.5,
                device=self.device,
                dtype=inputs_embed.dtype,
            )
        )
        context_weight = self.weights.tensor(
            "model.per_layer_model_projection.weight"
        )
        if self.kernel_backend is not None and self.gemma4_triton_ple:
            assert self._gemma4_context_buffer is not None
            context = self._runtime_matvec(
                context_weight,
                inputs_embed,
                self._gemma4_context_buffer,
            )
        else:
            context = torch.mv(context_weight, inputs_embed)
        context.mul_(self.hidden_size**-0.5)
        context = context.reshape(
            self.layers,
            self.descriptor.hidden_size_per_layer_input,
        )
        norm_weight = self.weights.tensor(
            "model.per_layer_projection_norm.weight"
        )
        work = context.float()
        context = (
            work
            * torch.rsqrt(
                work.pow(2).mean(dim=-1, keepdim=True)
                + self.norm_eps
            )
            * norm_weight.float()
        ).to(dtype=inputs_embed.dtype)
        return (context + token_component) * (2.0**-0.5)

    def _gemma4_attention(
        self,
        layer: int,
        hidden: torch.Tensor,
        token_index: int,
    ) -> torch.Tensor:
        layer_type = self.descriptor.layer_types[layer]
        is_full = layer_type == "full_attention"
        head_dim = (
            self.descriptor.global_head_dim
            if is_full
            else self.descriptor.head_dim
        )
        q_weight = self.weights.tensor(
            _layer_tensor(layer, "self_attn.q_proj.weight")
        )
        first_shared = (
            self.layers - self.descriptor.num_kv_shared_layers
        )
        if (
            self.kernel_backend is not None 
            and self.gemma4_triton_attention
            and q_weight.device.type == "cuda"
            and self._weight_scale(q_weight) is None
        ):
            assert self._gemma4_qkv_buffer is not None
            if layer < first_shared:
                k_weight = self.weights.tensor(
                    _layer_tensor(
                        layer, "self_attn.k_proj.weight"
                    )
                )
                v_weight = self.weights.tensor(
                    _layer_tensor(
                        layer, "self_attn.v_proj.weight"
                    )
                )
                rows = (
                    int(q_weight.shape[0]),
                    int(k_weight.shape[0]),
                    int(v_weight.shape[0]),
                )
                qkv = self._gemma4_qkv_buffer[: sum(rows)]
                self.kernel_backend.multi_matvec(
                    (q_weight, k_weight, v_weight),
                    hidden,
                    qkv,
                )
                q_raw = qkv[: rows[0]]
                k_raw = qkv[
                    rows[0] : rows[0] + rows[1]
                ]
                v_raw = qkv[rows[0] + rows[1] : sum(rows)]
            else:
                q_raw = self._runtime_matvec(
                    q_weight,
                    hidden,
                    self._gemma4_qkv_buffer[
                        : int(q_weight.shape[0])
                    ],
                )
                k_raw = None
                v_raw = None
        else:
            if getattr(self, "_gemma4_qkv_buffer", None) is None:
                q_raw = self._runtime_matvec(q_weight, hidden, torch.empty(q_weight.shape[0], device=hidden.device, dtype=hidden.dtype))
            else:
                q_raw = self._runtime_matvec(q_weight, hidden, self._gemma4_qkv_buffer[:int(q_weight.shape[0])])
            k_raw = None
            v_raw = None
        q = q_raw.reshape(self.heads, head_dim)
        q = self._gemma4_head_norm(
            q,
            self.weights.tensor(
                _layer_tensor(layer, "self_attn.q_norm.weight")
            ),
        )
        q = self._gemma4_rope(
            q, token_index, is_full=is_full
        )

        if layer < first_shared:
            if k_raw is None or v_raw is None:
                k_weight = self.weights.tensor(_layer_tensor(layer, "self_attn.k_proj.weight"))
                v_weight = self.weights.tensor(_layer_tensor(layer, "self_attn.v_proj.weight"))
                if getattr(self, "_gemma4_qkv_buffer", None) is None:
                    k_out = torch.empty(k_weight.shape[0], device=hidden.device, dtype=hidden.dtype)
                    v_out = torch.empty(v_weight.shape[0], device=hidden.device, dtype=hidden.dtype)
                else:
                    k_out = self._gemma4_qkv_buffer[int(q_weight.shape[0]) : int(q_weight.shape[0]) + int(k_weight.shape[0])]
                    v_out = self._gemma4_qkv_buffer[int(q_weight.shape[0]) + int(k_weight.shape[0]) : int(q_weight.shape[0]) + int(k_weight.shape[0]) + int(v_weight.shape[0])]
                k = self._runtime_matvec(k_weight, hidden, k_out).reshape(-1, head_dim)
                v = self._runtime_matvec(v_weight, hidden, v_out).reshape(-1, head_dim)
            else:
                k = k_raw.reshape(-1, head_dim)
                v = v_raw.reshape(-1, head_dim)
            k = self._gemma4_head_norm(
                k,
                self.weights.tensor(
                    _layer_tensor(
                        layer, "self_attn.k_norm.weight"
                    )
                ),
            )
            v_work = v.float()
            v = (
                v_work
                * torch.rsqrt(
                    v_work.pow(2).mean(
                        dim=-1, keepdim=True
                    )
                    + self.norm_eps
                )
            ).to(dtype=hidden.dtype)
            k = self._gemma4_rope(
                k, token_index, is_full=is_full
            )
            self._gemma4_key_history[layer].append(k)
            self._gemma4_value_history[layer].append(v)
            source_layer = layer
        else:
            source_layer = self._gemma4_shared_source[layer_type]

        keys = torch.stack(
            self._gemma4_key_history[source_layer], dim=2
        )
        values = torch.stack(
            self._gemma4_value_history[source_layer], dim=1
        )
        if not is_full and keys.shape[2] > 512:
            keys = keys[..., -512:]
            values = values[:, -512:]
        self._last_kv_tokens_attended = int(keys.shape[2])
        group = self.heads // int(keys.shape[0])
        query = q.reshape(int(keys.shape[0]), group, head_dim)
        scores = torch.matmul(query, keys)
        probabilities = torch.softmax(
            scores, dim=-1, dtype=torch.float32
        ).to(dtype=hidden.dtype)
        mixed = torch.matmul(probabilities, values).reshape(-1)
        o_weight = self.weights.tensor(
            _layer_tensor(layer, "self_attn.o_proj.weight")
        )
        if self._gemma4_projection_buffer is not None:
            return self._runtime_matvec(o_weight, mixed, out=self._gemma4_projection_buffer)
        return self._runtime_matvec(o_weight, mixed)

    def _gemma4_head_norm(
        self,
        value: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        work = value.float()
        return (
            work
            * torch.pow(
                work.pow(2).mean(dim=-1, keepdim=True)
                + self.norm_eps,
                -0.5,
            )
            * weight.float()
        ).to(dtype=value.dtype)

    def _gemma4_rope(
        self,
        value: torch.Tensor,
        token_index: int,
        *,
        is_full: bool,
    ) -> torch.Tensor:
        head_dim = int(value.shape[-1])
        if is_full:
            active_angles = int(0.25 * head_dim // 2)
            active = 2 * active_angles
            freq_index = torch.arange(
                0, active, 2, device=self.device, dtype=torch.float32
            )
            inv = 1_000_000.0 ** (-freq_index / head_dim)
            inv = torch.cat(
                (
                    inv,
                    torch.zeros(
                        head_dim // 2 - active_angles,
                        device=self.device,
                        dtype=torch.float32,
                    ),
                )
            )
        else:
            freq_index = torch.arange(
                0,
                head_dim,
                2,
                device=self.device,
                dtype=torch.float32,
            )
            inv = 10_000.0 ** (-freq_index / head_dim)
        frequencies = inv * float(token_index)
        cos = torch.cat((frequencies, frequencies)).cos().to(
            dtype=value.dtype
        )
        sin = torch.cat((frequencies, frequencies)).sin().to(
            dtype=value.dtype
        )
        first, second = value.chunk(2, dim=-1)
        rotated = torch.cat((-second, first), dim=-1)
        return value * cos + rotated * sin

    @torch.inference_mode()
    def _forward_token_qwen3_5(
        self,
        embed: torch.Tensor,
        token_id: int | torch.Tensor,
        layer_count: int,
        token_index: int,
    ) -> torch.Tensor:
        self._active_token_index = token_index
        if token_index == 0:
            for state in self._qwen35_conv_states.values():
                state.zero_()
            for state in self._qwen35_recurrent_states.values():
                state.zero_()

        if self._embed_scale is not None or (
            isinstance(self.weights, ThinGpuPagePool)
            and self.weights.cpu_embed
        ):
            if self.kernel_backend is not None:
                hidden = self._copy_embed_token(
                    embed,
                    token_id,
                    self.kernel_backend.buffers.hidden_a,
                )
            else:
                hidden = self._copy_embed_token(
                    embed,
                    token_id,
                    torch.empty(
                        self.hidden_size,
                        device=self.device,
                        dtype=self._final_norm_weight.dtype,
                    ),
                )
        elif isinstance(token_id, torch.Tensor):
            token = token_id.reshape(1).to(
                device=embed.device, dtype=torch.long
            )
            hidden = embed.index_select(0, token).reshape(-1).clone()
        else:
            hidden = embed[token_id].clone()

        for layer in range(layer_count):
            active_layer_pages = []
            fused_parents = []
            if isinstance(self.weights, ThinGpuPagePool):
                active_layer_pages = getattr(self.weights, "_precomputed_active_layer_pages", {}).get(layer)
                fused_parents = getattr(self.weights, "_precomputed_fused_parents", {}).get(layer)
                if active_layer_pages is None:
                    active_layer_pages = [
                        page_id for page_id in self.weights.page_specs.keys()
                        if f"model.layers.{layer}." in page_id
                    ]
                    fused_parents = [
                        self.weights.page_specs[p]["fused_to"]
                        for p in active_layer_pages
                        if self.weights.page_specs[p].get("fused_to") is not None
                    ]
                if active_layer_pages:
                    self.weights._active_pages.update(active_layer_pages)
                if fused_parents:
                    self.weights._active_pages.update(fused_parents)

            try:
                self._capture_debug(layer, "layer_input", hidden)
                for distance in range(1, self.prefetch_distance + 1):
                    if hasattr(self.weights, "prefetch_layer"):
                        self.weights.prefetch_layer(layer + distance)

                normed = self._normalization(
                    hidden,
                    _layer_tensor(layer, "input_layernorm.weight"),
                )
                self._capture_debug(layer, "post_input_rmsnorm", normed)
                layer_type = self.descriptor.layer_types[layer]
                if layer_type == "linear_attention":
                    mixed = self._qwen35_linear_attention(layer, normed)
                elif layer_type == "full_attention":
                    mixed = self._qwen35_full_attention(
                        layer, normed, token_index
                    )
                else:
                    raise RuntimeError(
                        f"unsupported Qwen3.5 layer type {layer_type!r}"
                    )
                hidden = hidden + mixed
                self._capture_debug(
                    layer, "post_attention_residual", hidden
                )

                mlp_input = self._normalization(
                    hidden,
                    _layer_tensor(
                        layer, "post_attention_layernorm.weight"
                    ),
                )
                gate_weight = self.weights.tensor(
                    _layer_tensor(layer, "mlp.gate_proj.weight")
                )
                up_weight = self.weights.tensor(
                    _layer_tensor(layer, "mlp.up_proj.weight")
                )
                down_weight = self.weights.tensor(
                    _layer_tensor(layer, "mlp.down_proj.weight")
                )
                if self.kernel_backend is not None:
                    buffers = self.kernel_backend.buffers
                    self._qwen35_multi_matvec(
                        (gate_weight, up_weight),
                        mlp_input,
                        buffers.gate_up,
                    )
                    gate = buffers.gate_up[: self.intermediate_size]
                    up = buffers.gate_up[self.intermediate_size :]
                    activation = self._runtime_silu_mul(
                        gate, up, buffers.mlp_act
                    )
                    mlp = self._runtime_matvec(
                        down_weight, activation, buffers.mlp
                    )
                else:
                    gate = torch.mv(gate_weight, mlp_input)
                    up = torch.mv(up_weight, mlp_input)
                    activation = torch.nn.functional.silu(gate) * up
                    mlp = torch.mv(down_weight, activation)
                hidden = hidden + mlp
                self._capture_debug(
                    layer, "final_residual_after_mlp", hidden
                )
                if self.evict_completed_layers and hasattr(
                    self.weights, "evict_completed_layer"
                ):
                    self.weights.evict_completed_layer(layer)
            finally:
                if isinstance(self.weights, ThinGpuPagePool):
                    if active_layer_pages:
                        self.weights._active_pages.difference_update(active_layer_pages)
                    if fused_parents:
                        self.weights._active_pages.difference_update(fused_parents)

        return self._normalization(hidden, "model.norm.weight")

    def _qwen35_multi_matvec(
        self,
        weights: tuple[torch.Tensor, ...],
        hidden: torch.Tensor,
        out: torch.Tensor,
    ) -> torch.Tensor:
        backend = self.kernel_backend
        assert backend is not None
        if any(weight.device.type == "cpu" for weight in weights):
            if (
                isinstance(self.weights, ThinGpuPagePool)
                and all(weight.device.type == "cpu" for weight in weights)
            ):
                page_ids = tuple(
                    self.weights._tensor_id_to_page_id.get(id(weight), "")
                    for weight in weights
                )
                if all(page_id in self.weights._cpu_int4_pages for page_id in page_ids):
                    return self.weights.cpu_int4_multi_matvec(
                        page_ids, hidden, out
                    )
            offset = 0
            for weight in weights:
                rows = int(weight.shape[0])
                self._runtime_matvec(
                    weight,
                    hidden,
                    out[offset : offset + rows],
                )
                offset += rows
            return out
        scales = tuple(self._weight_scale(weight) for weight in weights)
        if (
            len(weights) == 2
            and isinstance(self.weights, ThinGpuPagePool)
            and all(scale is not None for scale in scales)
        ):
            page_ids = tuple(
                self.weights._tensor_id_to_page_id.get(id(weight), "")
                for weight in weights
            )
            metadata = tuple(
                self.weights._mxfp4_metadata_by_page.get(page_id)
                for page_id in page_ids
            )
            if (
                all(item is not None for item in metadata)
                and metadata[0][:2] == metadata[1][:2]
                and not any(
                    page_id in getattr(
                        self.weights, "_mxfp4_binary_residuals_by_page", {}
                    )
                    for page_id in page_ids
                )
                and not any(
                    page_id in getattr(
                        self.weights, "_mxfp4_row_postscales_by_page", {}
                    )
                    for page_id in page_ids
                )
            ):
                rows, cols = metadata[0][:2]
                result = backend.dual_mxfp4_tensorcore_matvec(
                    (weights[0], weights[1]),
                    (scales[0], scales[1]),
                    hidden,
                    out,
                    rows=int(rows),
                    cols=int(cols),
                )
                self._record_streamed_tensor_use(*weights)
                return result
        if all(scale is None for scale in scales):
            large_rows = max(int(weight.shape[0]) for weight in weights)
            result = backend.multi_matvec(
                weights,
                hidden,
                out,
                block_m=64 if large_rows >= 512 else 8,
                num_warps=4,
            )
            self._record_streamed_tensor_use(*weights)
            return result
        offset = 0
        for weight in weights:
            rows_i = int(weight.shape[0])
            self._runtime_matvec(weight, hidden, out[offset : offset + rows_i])
            offset += rows_i
        return out

    def _qwen35_linear_attention(
        self,
        layer: int,
        hidden: torch.Tensor,
    ) -> torch.Tensor:
        prefix = "linear_attn"
        qkv_weight = self.weights.tensor(
            _layer_tensor(layer, f"{prefix}.in_proj_qkv.weight")
        )
        z_weight = self.weights.tensor(
            _layer_tensor(layer, f"{prefix}.in_proj_z.weight")
        )
        a_weight = self.weights.tensor(
            _layer_tensor(layer, f"{prefix}.in_proj_a.weight")
        )
        b_weight = self.weights.tensor(
            _layer_tensor(layer, f"{prefix}.in_proj_b.weight")
        )
        if self.kernel_backend is not None:
            assert self._qwen35_projection_buffer is not None
            projection = self._qwen35_projection_buffer
            qkv_rows = int(qkv_weight.shape[0])
            z_rows = int(z_weight.shape[0])
            total_rows = (
                qkv_rows + z_rows
                + int(a_weight.shape[0]) + int(b_weight.shape[0])
            )
            self._qwen35_multi_matvec(
                (qkv_weight, z_weight, a_weight, b_weight),
                hidden,
                projection[:total_rows],
            )
            mixed_qkv = projection[:qkv_rows]
            z = projection[qkv_rows : qkv_rows + z_rows]
            a_start = qkv_rows + z_rows
            a = projection[a_start : a_start + int(a_weight.shape[0])]
            b = projection[
                a_start + int(a_weight.shape[0]) :
                a_start + int(a_weight.shape[0]) + int(b_weight.shape[0])
            ]
        else:
            mixed_qkv = torch.mv(qkv_weight, hidden)
            z = torch.mv(z_weight, hidden)
            a = torch.mv(a_weight, hidden)
            b = torch.mv(b_weight, hidden)

        conv_state = self._qwen35_conv_states[layer]
        conv_weight = self.weights.tensor(
            _layer_tensor(layer, f"{prefix}.conv1d.weight")
        )
        a_log = self.weights.tensor(
            _layer_tensor(layer, f"{prefix}.A_log")
        )
        dt_bias = self.weights.tensor(
            _layer_tensor(layer, f"{prefix}.dt_bias")
        )
        norm_weight = self.weights.tensor(
            _layer_tensor(layer, f"{prefix}.norm.weight")
        )
        if self.kernel_backend is not None:
            assert self._qwen35_core_buffer is not None
            gated = self.kernel_backend.qwen35_gated_deltanet(
                mixed_qkv,
                z,
                a,
                b,
                conv_weight,
                dt_bias,
                a_log,
                norm_weight,
                conv_state,
                self._qwen35_recurrent_states[layer],
                self._qwen35_core_buffer,
                key_heads=self.descriptor.linear_num_key_heads,
                value_heads=self.descriptor.linear_num_value_heads,
                key_dim=self.descriptor.linear_key_head_dim,
                value_dim=self.descriptor.linear_value_head_dim,
                eps=self.norm_eps,
            )
            self._record_streamed_tensor_use(
                conv_weight,
                dt_bias,
                a_log,
                norm_weight,
            )
            out_weight = self.weights.tensor(
                _layer_tensor(layer, f"{prefix}.out_proj.weight")
            )
            return self._runtime_matvec(
                out_weight,
                gated,
                self.kernel_backend.buffers.attn_out,
            )

        conv_state.copy_(
            torch.cat((conv_state[:, 1:], mixed_qkv[:, None]), dim=1)
        )
        convolved = torch.nn.functional.conv1d(
            conv_state.unsqueeze(0),
            conv_weight,
            groups=int(conv_state.shape[0]),
        ).reshape(-1)
        convolved = torch.nn.functional.silu(convolved)

        key_dim = (
            self.descriptor.linear_num_key_heads
            * self.descriptor.linear_key_head_dim
        )
        value_dim = (
            self.descriptor.linear_num_value_heads
            * self.descriptor.linear_value_head_dim
        )
        query, key, value = torch.split(
            convolved, (key_dim, key_dim, value_dim)
        )
        query = query.reshape(
            self.descriptor.linear_num_key_heads,
            self.descriptor.linear_key_head_dim,
        )
        key = key.reshape(
            self.descriptor.linear_num_key_heads,
            self.descriptor.linear_key_head_dim,
        )
        value = value.reshape(
            self.descriptor.linear_num_value_heads,
            self.descriptor.linear_value_head_dim,
        )

        query = query * torch.rsqrt(
            (query * query).sum(dim=-1, keepdim=True) + 1e-6
        )
        key = key * torch.rsqrt(
            (key * key).sum(dim=-1, keepdim=True) + 1e-6
        )
        repeats = (
            self.descriptor.linear_num_value_heads
            // self.descriptor.linear_num_key_heads
        )
        if repeats > 1:
            query = query.repeat_interleave(repeats, dim=0)
            key = key.repeat_interleave(repeats, dim=0)

        beta = torch.sigmoid(b).float()
        g = -a_log.float().exp() * torch.nn.functional.softplus(
            a.float() + dt_bias.float()
        )

        recurrent = self._qwen35_recurrent_states[layer]
        recurrent.mul_(torch.exp(g)[:, None, None])
        query_f = query.float() * (
            self.descriptor.linear_key_head_dim ** -0.5
        )
        key_f = key.float()
        value_f = value.float()
        kv_mem = (recurrent * key_f.unsqueeze(-1)).sum(dim=-2)
        delta = (value_f - kv_mem) * beta[:, None]
        recurrent.add_(key_f.unsqueeze(-1) * delta.unsqueeze(-2))
        output = (
            recurrent * query_f.unsqueeze(-1)
        ).sum(dim=-2).to(dtype=hidden.dtype)

        output_2d = output.reshape(
            self.descriptor.linear_num_value_heads,
            self.descriptor.linear_value_head_dim,
        )
        z_2d = z.reshape_as(output_2d)
        work = output_2d.float()
        work = work * torch.rsqrt(
            work.pow(2).mean(dim=-1, keepdim=True) + self.norm_eps
        )
        gated = norm_weight * work.to(dtype=hidden.dtype)
        gated = (
            gated * torch.nn.functional.silu(z_2d.float())
        ).to(dtype=hidden.dtype)
        out_weight = self.weights.tensor(
            _layer_tensor(layer, f"{prefix}.out_proj.weight")
        )
        if self.kernel_backend is not None:
            return self._runtime_matvec(
                out_weight,
                gated.reshape(-1),
                self.kernel_backend.buffers.attn_out,
            )
        return torch.mv(out_weight, gated.reshape(-1))

    def _qwen35_full_attention(
        self,
        layer: int,
        hidden: torch.Tensor,
        token_index: int,
    ) -> torch.Tensor:
        q_weight = self.weights.tensor(
            _layer_tensor(layer, "self_attn.q_proj.weight")
        )
        k_weight = self.weights.tensor(
            _layer_tensor(layer, "self_attn.k_proj.weight")
        )
        v_weight = self.weights.tensor(
            _layer_tensor(layer, "self_attn.v_proj.weight")
        )
        if self.kernel_backend is not None:
            assert self._qwen35_projection_buffer is not None
            projection = self._qwen35_projection_buffer
            rows = (
                int(q_weight.shape[0]),
                int(k_weight.shape[0]),
                int(v_weight.shape[0]),
            )
            self._qwen35_multi_matvec(
                (q_weight, k_weight, v_weight),
                hidden,
                projection[: sum(rows)],
            )
            q_raw = projection[: rows[0]]
            k_raw = projection[rows[0] : rows[0] + rows[1]]
            v = projection[rows[0] + rows[1] : sum(rows)]
        else:
            q_raw = torch.mv(q_weight, hidden)
            k_raw = torch.mv(k_weight, hidden)
            v = torch.mv(v_weight, hidden)
        q_projected = q_raw.reshape(self.heads, self.head_dim * 2)
        q, gate = q_projected.chunk(2, dim=-1)
        k = k_raw.reshape(self.kv_heads, self.head_dim)
        q_norm = self.weights.tensor(
            _layer_tensor(layer, "self_attn.q_norm.weight")
        )
        k_norm = self.weights.tensor(
            _layer_tensor(layer, "self_attn.k_norm.weight")
        )
        q_work = q.float()
        q = (
            q_work
            * torch.rsqrt(
                q_work.pow(2).mean(dim=-1, keepdim=True)
                + self.norm_eps
            )
            * (1.0 + q_norm.float())
        ).to(dtype=hidden.dtype)
        k_work = k.float()
        k = (
            k_work
            * torch.rsqrt(
                k_work.pow(2).mean(dim=-1, keepdim=True)
                + self.norm_eps
            )
            * (1.0 + k_norm.float())
        ).to(dtype=hidden.dtype)
        q, k = self._apply_rope(
            q.reshape(-1), k.reshape(-1), layer, token_index
        )
        if self.kv_cache is not None and not getattr(
            self, "_autotuning", False
        ):
            self.kv_cache.append(layer, k, v, token_index)
        attended = self._attention(q, k, v, layer, token_index)
        attended.mul_(torch.sigmoid(gate.reshape(-1)))
        o_weight = self.weights.tensor(
            _layer_tensor(layer, "self_attn.o_proj.weight")
        )
        if self.kernel_backend is not None:
            return self._runtime_matvec(
                o_weight,
                attended,
                self.kernel_backend.buffers.attn_out,
            )
        return torch.mv(o_weight, attended)

    @torch.inference_mode()
    def forward_token_debug(
        self,
        token_id: int | torch.Tensor,
        *,
        layer: int,
        token_index: int,
    ) -> dict[str, torch.Tensor]:
        """Debug-only component capture; never used by timed decode."""
        if layer < 0 or layer >= self.layers:
            raise ValueError(f"debug layer {layer} outside 0..{self.layers - 1}")
        previous_layer = self._debug_layer
        previous_components = self._debug_components
        self._debug_layer = layer
        self._debug_components = {}
        try:
            hidden = self.forward_token(token_id, token_index=token_index)
            self._debug_components["final_norm"] = hidden.detach().clone()
            self._debug_components["final_logits"] = (
                self.logits(hidden).detach().clone()
            )
            return dict(self._debug_components)
        finally:
            self._debug_layer = previous_layer
            if previous_layer is None:
                self._debug_components = {}
            else:
                self._debug_components = previous_components

    @torch.inference_mode()
    def forward_token_debug_all(
        self,
        token_id: int | torch.Tensor,
        *,
        token_index: int,
    ) -> dict[str, torch.Tensor]:
        """Capture every layer for first-divergence analysis outside benchmarks."""
        previous_layer = self._debug_layer
        previous_all_layers = self._debug_all_layers
        previous_components = self._debug_components
        self._debug_layer = None
        self._debug_all_layers = True
        self._debug_components = {}
        try:
            hidden = self.forward_token(token_id, token_index=token_index)
            self._debug_components["final_norm"] = hidden.detach().clone()
            self._debug_components["final_logits"] = (
                self.logits(hidden).detach().clone()
            )
            return dict(self._debug_components)
        finally:
            self._debug_layer = previous_layer
            self._debug_all_layers = previous_all_layers
            if previous_layer is None and not previous_all_layers:
                self._debug_components = {}
            else:
                self._debug_components = previous_components

    def _capture_debug(
        self,
        layer: int,
        name: str,
        tensor: torch.Tensor,
    ) -> None:
        if self._debug_all_layers:
            self._debug_components[
                f"layer_{layer}.{name}"
            ] = tensor.detach().clone()
        elif self._debug_layer == layer:
            self._debug_components[name] = tensor.detach().clone()

    def _apply_rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        layer: int,
        token_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.descriptor.layer_uses_rope(layer):
            return q, k
        rotary_dim = int(self.head_dim * self.partial_rotary_factor)
        if rotary_dim <= 0 or rotary_dim > self.head_dim or rotary_dim % 2:
            raise RuntimeError(
                "RoPE requires a positive even rotary dimension no larger "
                f"than head_dim, got rotary_dim={rotary_dim}, "
                f"head_dim={self.head_dim}"
            )
        half = rotary_dim // 2
        cos, sin = self._rope_values(token_index, q.dtype)
        if (
            rotary_dim == self.head_dim
            and self.fused_rope_enabled
            and self.kernel_backend is not None
            and cos.ndim == 1
        ):
            return self.kernel_backend.rope_qk_inplace(
                q,
                k,
                cos,
                sin,
                self.heads,
                self.kv_heads,
                self.head_dim,
            )

        q_heads = q.reshape(self.heads, self.head_dim)
        k_heads = k.reshape(self.kv_heads, self.head_dim)
        if rotary_dim == self.head_dim:
            q_first, q_second = q_heads[:, :half], q_heads[:, half:]
            k_first, k_second = k_heads[:, :half], k_heads[:, half:]
            q_first_copy = q_first.clone()
            q_first.mul_(cos).sub_(q_second * sin)
            q_second.mul_(cos).add_(q_first_copy * sin)
            k_first_copy = k_first.clone()
            k_first.mul_(cos).sub_(k_second * sin)
            k_second.mul_(cos).add_(k_first_copy * sin)
            return q, k

        q_rot, q_pass = q_heads[:, :rotary_dim], q_heads[:, rotary_dim:]
        k_rot, k_pass = k_heads[:, :rotary_dim], k_heads[:, rotary_dim:]
        q_first, q_second = q_rot[:, :half], q_rot[:, half:]
        k_first, k_second = k_rot[:, :half], k_rot[:, half:]
        q_rotated = torch.cat(
            (
                q_first * cos - q_second * sin,
                q_second * cos + q_first * sin,
            ),
            dim=-1,
        )
        k_rotated = torch.cat(
            (
                k_first * cos - k_second * sin,
                k_second * cos + k_first * sin,
            ),
            dim=-1,
        )
        q = torch.cat((q_rotated, q_pass), dim=-1).reshape(-1)
        k = torch.cat((k_rotated, k_pass), dim=-1).reshape(-1)
        return q, k

    def _rope_values(
        self,
        token_index: int,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cached = self._rope_cos_sin_cache.get(token_index)
        if cached is None or cached[0].dtype != dtype:
            inv_freq, attention_scaling = self._rope_parameters(
                token_index + 1
            )
            frequencies = inv_freq * float(token_index)
            cached = (
                (frequencies.cos() * attention_scaling).to(dtype=dtype),
                (frequencies.sin() * attention_scaling).to(dtype=dtype),
            )
            self._rope_cos_sin_cache[token_index] = cached
        return cached

    def _rope_parameters(
        self,
        seq_len: int,
    ) -> tuple[torch.Tensor, float]:
        variant = self.descriptor.rope_variant
        rotary_dim = int(self.head_dim * self.partial_rotary_factor)
        if variant in {None, "", "default"}:
            if self._rope_inv_freq is None:
                freq_index = torch.arange(
                    0,
                    rotary_dim,
                    2,
                    device=self.device,
                    dtype=torch.float32,
                )
                self._rope_inv_freq = self.descriptor.rope_theta ** (
                    -freq_index / rotary_dim
                )
            return self._rope_inv_freq, 1.0
        dynamic = variant in {"dynamic", "longrope"}
        if (
            self._rope_inv_freq is not None
            and (not dynamic or self._rope_dynamic_seq_len == seq_len)
        ):
            return self._rope_inv_freq, self._rope_attention_scaling
        try:
            from transformers import PreTrainedConfig
            from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
        except ImportError as exc:
            raise RuntimeError(
                f"RoPE variant {variant!r} requires Transformers"
            ) from exc
        initializer = ROPE_INIT_FUNCTIONS.get(variant)
        if initializer is None:
            raise RuntimeError(f"unsupported RoPE variant {variant!r}")
        rope_parameters = dict(
            self.descriptor.rope_parameters
            or self.descriptor.rope_scaling
            or {}
        )
        rope_parameters.setdefault("rope_type", variant)
        rope_parameters.setdefault("rope_theta", self.descriptor.rope_theta)
        config = PreTrainedConfig()
        config.rope_parameters = rope_parameters
        config.head_dim = rotary_dim
        config.hidden_size = self.hidden_size
        config.num_attention_heads = self.heads
        config.max_position_embeddings = (
            self.descriptor.max_position_embeddings or seq_len
        )
        if self.descriptor.original_max_position_embeddings is not None:
            config.original_max_position_embeddings = (
                self.descriptor.original_max_position_embeddings
            )
        inv_freq, scaling = initializer(
            config,
            device=self.device,
            seq_len=seq_len,
        )
        self._rope_inv_freq = inv_freq
        self._rope_attention_scaling = float(scaling)
        self._rope_dynamic_seq_len = seq_len if dynamic else None
        return self._rope_inv_freq, self._rope_attention_scaling

    def _attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: int,
        token_index: int,
        lev: Optional[LayerProfileEvents] = None,
    ) -> torch.Tensor:
        if self.attention_mode == "current_only_smoke" or getattr(
            self, "_autotuning", False
        ):
            self._last_kv_tokens_attended = 1
            return _single_token_attention(
                q, k, v, self.heads, self.kv_heads, self.head_dim
            )
        if self.kv_cache is None:
            raise RuntimeError("causal_kv attention requires a PagedKVCache")

        window = self.descriptor.layer_attention_window(layer)
        start_token = (
            max(0, token_index - window + 1) if window is not None else 0
        )
        keys, values = self.kv_cache.history(
            layer,
            token_index,
            start_token=start_token,
        )
        self._last_kv_tokens_attended = int(keys.shape[1])
        sinks = self._bound_layer_tensor(layer, "attention_sinks")
        if sinks is None:
            sinks = _optional_tensor(
                self.weights,
                _layer_tensor(layer, "self_attn.sinks"),
            )
        if (
            self.attention_backend == "sdpa"
            and self.attention_logit_softcap is None
            and sinks is None
            and self._debug_layer is None
            and not self._debug_all_layers
        ):
            if lev is not None:
                lev.qk_start.record()
            mixed = torch.nn.functional.scaled_dot_product_attention(
                q.reshape(1, self.heads, 1, self.head_dim),
                keys.unsqueeze(0),
                values.unsqueeze(0),
                dropout_p=0.0,
                is_causal=False,
                enable_gqa=self.heads != self.kv_heads,
            )
            if lev is not None:
                lev.qk_end.record()
            return mixed.reshape(self.heads * self.head_dim)
        if (
            self.attention_backend in {"triton_fused", "triton_split"}
            and self.attention_logit_softcap is None
            and self.kernel_backend is not None
            and (
                self._adaptive_switch_token_index < 0
                or token_index >= self._adaptive_switch_token_index
            )
            and (sinks is None or self.attention_backend == "triton_fused")
            and self.head_dim <= 256
            and self._debug_layer is None
            and not self._debug_all_layers
        ):
            if lev is not None:
                lev.qk_start.record()
            if self.attention_backend == "triton_split":
                mixed = (
                    self.kernel_backend.split_single_token_gqa_attention(
                        q,
                        keys,
                        values,
                        self.kernel_backend.buffers.attn,
                        self.heads,
                        self.kv_heads,
                        self.head_dim,
                    )
                )
            else:
                mixed = self.kernel_backend.single_token_gqa_attention(
                    q,
                    keys,
                    values,
                    self.kernel_backend.buffers.attn,
                    self.heads,
                    self.kv_heads,
                    self.head_dim,
                    sinks=sinks,
                )
            if lev is not None:
                lev.qk_end.record()
            return mixed
        group = self.heads // self.kv_heads
        query = q.reshape(self.kv_heads, group, self.head_dim)

        if lev is not None:
            lev.qk_start.record()
        scores = torch.matmul(query, keys.transpose(1, 2))
        scores.mul_(
            self.attention_scale
            if self.attention_scale is not None
            else self.head_dim**-0.5
        )
        if self.attention_logit_softcap is not None:
            scores.div_(self.attention_logit_softcap)
            scores.tanh_()
            scores.mul_(self.attention_logit_softcap)
        self._capture_debug(layer, "attention_scores", scores)
        if lev is not None:
            lev.qk_end.record()
            lev.softmax_start.record()
        if sinks is not None:
            combined = torch.cat(
                (
                    scores,
                    sinks.reshape(self.kv_heads, group, 1).to(
                        dtype=scores.dtype
                    ),
                ),
                dim=-1,
            )
            combined.sub_(combined.max(dim=-1, keepdim=True).values)
            probabilities = torch.softmax(
                combined,
                dim=-1,
                dtype=combined.dtype,
            )[..., :-1].to(dtype=query.dtype)
        else:
            probabilities = torch.softmax(
                scores, dim=-1, dtype=torch.float32
            ).to(dtype=query.dtype)
        self._capture_debug(layer, "attention_probs", probabilities)
        if lev is not None:
            lev.softmax_end.record()
            lev.value_mix_start.record()
        mixed = torch.matmul(probabilities, values)
        self._capture_debug(
            layer,
            "attention_output_before_o_proj",
            mixed.reshape(self.heads * self.head_dim),
        )
        if lev is not None:
            lev.value_mix_end.record()
        return mixed.reshape(self.heads * self.head_dim)

    @torch.inference_mode()
    def _forward_token_moe(
        self,
        embed: torch.Tensor,
        token_id: int | torch.Tensor,
        layer_count: int,
        token_index: int,
    ) -> torch.Tensor:
        if getattr(self, "_moe_qkv_buffer", None) is None:
            self._moe_qkv_buffer = torch.empty(
                (int(self.q_dim + 2 * self.kv_dim),),
                device=self.device,
                dtype=embed.dtype,
            )
            self._moe_attention_output_buffer = torch.empty(
                (self.hidden_size,), device=self.device, dtype=embed.dtype
            )
        if self._embed_scale is not None:
            hidden = self._copy_embed_token(
                embed,
                token_id,
                torch.empty(
                    self.hidden_size,
                    device=self.device,
                    dtype=self._final_norm_weight.dtype,
                ),
            )
        else:
            hidden = embed[token_id].clone()
        if self.embedding_scale != 1.0:
            hidden.mul_(
                torch.tensor(
                     self.embedding_scale,
                     device=hidden.device,
                     dtype=hidden.dtype,
                )
            )
        if getattr(self, "_moe_next_hidden_buffer", None) is None:
            self._moe_next_hidden_buffer = torch.empty_like(hidden)
            self._moe_normed_buffer = torch.empty_like(hidden)
        hidden_a = hidden
        hidden_b = self._moe_next_hidden_buffer
        normed = self._normalization(
            hidden_a,
            _layer_tensor(0, "input_layernorm.weight"),
        )
        for layer in range(layer_count):
            lev = None
            if getattr(self, "_profiler_enabled", False):
                lev = LayerProfileEvents()
                self._current_step_profile["layers"].append(lev)
                lev.layer_start.record()
                lev.attn_start.record()
                lev.qkv_start.record()
            self._capture_debug(layer, "layer_input", hidden_a)
            for distance in range(1, self.prefetch_distance + 1):
                if hasattr(self.weights, "prefetch_layer"):
                    self.weights.prefetch_layer(layer + distance)
            self._capture_debug(layer, "post_input_rmsnorm", normed)
            if getattr(self, "is_gemma4", False):
                if lev is not None:
                    lev.qkv_start.record()
                    lev.qkv_end.record()
                    lev.o_proj_start.record()
                attention_output = self._gemma4_attention(layer, normed, token_index)
                if lev is not None:
                    lev.o_proj_end.record()
            else:
                q_proj_id = _layer_tensor(layer, "self_attn.q_proj.weight")
                q_proj = self._bound_layer_tensor(layer, "q_weight")
                if q_proj is not None or q_proj_id in self.weights.page_specs:
                    if q_proj is None:
                        q_proj = self.weights.tensor(q_proj_id)
                    k_proj = self._bound_layer_tensor(layer, "k_weight")
                    if k_proj is None:
                        k_proj = self.weights.tensor(
                            _layer_tensor(layer, "self_attn.k_proj.weight")
                        )
                    v_proj = self._bound_layer_tensor(layer, "v_weight")
                    if v_proj is None:
                        v_proj = self.weights.tensor(
                            _layer_tensor(layer, "self_attn.v_proj.weight")
                        )
                    if self.kernel_backend is not None:
                        self.kernel_backend.multi_matvec(
                            (q_proj, k_proj, v_proj),
                            normed,
                            self._moe_qkv_buffer,
                            block_m=64,
                        )
                        q = self._moe_qkv_buffer[: self.q_dim]
                        k = self._moe_qkv_buffer[
                            self.q_dim : self.q_dim + self.kv_dim
                        ]
                        v = self._moe_qkv_buffer[
                            self.q_dim + self.kv_dim :
                        ]
                    else:
                        q = self._runtime_matvec(
                            q_proj, normed, self._moe_qkv_buffer[: self.q_dim]
                        )
                        k = self._runtime_matvec(
                            k_proj,
                            normed,
                            self._moe_qkv_buffer[
                                self.q_dim : self.q_dim + self.kv_dim
                            ],
                        )
                        v = self._runtime_matvec(
                            v_proj,
                            normed,
                            self._moe_qkv_buffer[self.q_dim + self.kv_dim :],
                        )
                else:
                    qkv_weight = _first_optional_tensor(
                        self.weights,
                        (
                            _layer_tensor(layer, "self_attn.qkv_proj.weight"),
                            _layer_tensor(layer, "attn.c_attn.weight"),
                        ),
                    )
                    if qkv_weight is None:
                        raise RuntimeError(f"layer {layer} has no supported QKV weights")
                    qkv_out = (
                        self._runtime_matvec(
                            qkv_weight, normed, self._moe_qkv_buffer
                        )
                        if self.kernel_backend is not None
                        else torch.mv(qkv_weight, normed)
                    )
                    q = qkv_out[:self.q_dim]
                    k = qkv_out[self.q_dim : self.q_dim + self.kv_dim]
                    v = qkv_out[self.q_dim + self.kv_dim :]
                if lev is not None:
                    lev.qkv_end.record()
                for projected, role, suffix in (
                    (q, "q_bias", "self_attn.q_proj.bias"),
                    (k, "k_bias", "self_attn.k_proj.bias"),
                    (v, "v_bias", "self_attn.v_proj.bias"),
                ):
                    bias = self._bound_layer_tensor(layer, role)
                    if bias is None:
                        bias = _optional_tensor(
                            self.weights,
                            _layer_tensor(layer, suffix),
                        )
                    if bias is not None:
                        projected.add_(bias)
                fused_qkv_bias = _optional_tensor(
                    self.weights,
                    _layer_tensor(layer, "self_attn.qkv_proj.bias"),
                )
                if fused_qkv_bias is not None:
                    q_bias, k_bias, v_bias = torch.split(
                        fused_qkv_bias,
                        (self.q_dim, self.kv_dim, self.kv_dim),
                    )
                    q.add_(q_bias)
                    k.add_(k_bias)
                    v.add_(v_bias)
                self._capture_debug(layer, "q_projection", q)
                self._capture_debug(layer, "k_projection", k)
                self._capture_debug(layer, "v_projection", v)
                q_norm = self._bound_layer_tensor(layer, "q_norm")
                if q_norm is None:
                    q_norm = _optional_tensor(
                        self.weights,
                        _layer_tensor(layer, "self_attn.q_norm.weight"),
                    )
                k_norm = self._bound_layer_tensor(layer, "k_norm")
                if k_norm is None:
                    k_norm = _optional_tensor(
                        self.weights,
                        _layer_tensor(layer, "self_attn.k_norm.weight"),
                    )
                fused_qk_norm_rope = (
                    q_norm is not None
                    and k_norm is not None
                    and q_norm.numel() == self.head_dim
                    and k_norm.numel() == self.head_dim
                    and self.attention_mode == "causal_kv"
                    and self.descriptor.layer_uses_rope(layer)
                    and int(self.head_dim * self.partial_rotary_factor)
                    == self.head_dim
                    and self.fused_rope_enabled
                    and self.kernel_backend is not None
                )
                if fused_qk_norm_rope:
                    cos, sin = self._rope_values(token_index, q.dtype)
                    q, k = self.kernel_backend.qk_head_rmsnorm_rope_inplace(
                        q,
                        k,
                        q_norm,
                        k_norm,
                        cos,
                        sin,
                        self.heads,
                        self.kv_heads,
                        self.head_dim,
                        self.rms_norm_eps,
                    )
                elif q_norm is not None:
                    q = (
                        _rms_norm(q, q_norm, self.rms_norm_eps)
                        if q_norm.numel() == q.numel()
                        else _head_rms_norm(
                            q,
                            q_norm,
                            self.heads,
                            self.head_dim,
                            self.rms_norm_eps,
                        )
                    )
                if k_norm is not None and not fused_qk_norm_rope:
                    k = (
                        _rms_norm(k, k_norm, self.rms_norm_eps)
                        if k_norm.numel() == k.numel()
                        else _head_rms_norm(
                            k,
                            k_norm,
                            self.kv_heads,
                            self.head_dim,
                            self.rms_norm_eps,
                        )
                    )
                if self.attention_mode == "causal_kv" and not fused_qk_norm_rope:
                    q, k = self._apply_rope(q, k, layer, token_index)
                if self.kv_cache is not None and not getattr(
                    self, "_autotuning", False
                ):
                    self.kv_cache.append(layer, k, v, token_index)
                attention = self._attention(
                    q,
                    k,
                    v,
                    layer,
                    token_index,
                    lev,
                )
                if lev is not None:
                    lev.o_proj_start.record()
                o_weight = self._bound_layer_tensor(layer, "o_weight")
                if o_weight is None:
                    o_weight = self.weights.tensor(
                        _layer_tensor(layer, "self_attn.o_proj.weight")
                    )
                attention_output = (
                    self._runtime_matvec(
                        o_weight,
                        attention,
                        self._moe_attention_output_buffer,
                    )
                    if self.kernel_backend is not None
                    else torch.mv(o_weight, attention)
                )
                o_bias = self._bound_layer_tensor(layer, "o_bias")
                if o_bias is None:
                    o_bias = _optional_tensor(
                        self.weights,
                        _layer_tensor(layer, "self_attn.o_proj.bias"),
                    )
                if o_bias is not None:
                    attention_output.add_(o_bias)
                if lev is not None:
                    lev.o_proj_end.record()

                if self.kernel_backend is not None:
                    post_attention_norm = self._bound_layer_tensor(
                        layer,
                        "post_attention_norm",
                    )
                    if post_attention_norm is None:
                        post_attention_norm = self.weights.tensor(
                            _layer_tensor(
                                layer,
                                "post_attention_layernorm.weight",
                            )
                        )
                    self.kernel_backend.add_rms_norm(
                        hidden_a,
                        attention_output,
                        hidden_b,
                        post_attention_norm,
                        self._moe_normed_buffer,
                        self.rms_norm_eps,
                    )
                    normed = self._moe_normed_buffer
                else:
                    hidden_b.copy_(hidden_a + attention_output)
                    normed = self._normalization(
                        hidden_b,
                        _layer_tensor(
                            layer,
                            "post_attention_layernorm.weight",
                        ),
                    )
                self._capture_debug(layer, "post_attention_residual", hidden_b)
                self._capture_debug(layer, "post_attention_rmsnorm", normed)
            if self.kernel_backend is not None:
                post_attention_norm = self._bound_layer_tensor(
                    layer,
                    "post_attention_norm",
                )
                if post_attention_norm is None:
                    post_attention_norm = self.weights.tensor(
                        _layer_tensor(
                            layer,
                            "post_attention_layernorm.weight",
                        )
                    )
                self.kernel_backend.add_rms_norm(
                    hidden_a,
                    attention_output,
                    hidden_b,
                    post_attention_norm,
                    self._moe_normed_buffer,
                    self.rms_norm_eps,
                )
                normed = self._moe_normed_buffer
            else:
                hidden_b.copy_(hidden_a + attention_output)
                normed = self._normalization(
                    hidden_b,
                    _layer_tensor(
                        layer,
                        "post_attention_layernorm.weight",
                    ),
                )
            self._capture_debug(layer, "post_attention_residual", hidden_b)
            self._capture_debug(layer, "post_attention_rmsnorm", normed)
            if lev is not None:
                lev.attn_end.record()
                lev.mlp_start.record()
            layer_graph = self.runtime_executor.layers[layer]
            if layer_graph.feedforward == "sparse_moe":
                moe = self._packed_moe_forward(layer, normed)
            elif layer_graph.feedforward == "gated_mlp":
                moe = self._dense_graph_mlp_forward(layer, normed)
            else:
                raise RuntimeError(
                    "unregistered feedforward block "
                    f"{layer_graph.feedforward!r} in layer {layer}"
                )
            
            if layer < layer_count - 1:
                if self.kernel_backend is not None:
                    next_input_norm = self._bound_layer_tensor(
                        layer + 1,
                        "input_norm",
                    )
                    if next_input_norm is None:
                        next_input_norm = self.weights.tensor(
                            _layer_tensor(
                                layer + 1,
                                "input_layernorm.weight",
                            )
                        )
                    self.kernel_backend.add_rms_norm(
                        hidden_b,
                        moe,
                        hidden_a,
                        next_input_norm,
                        self._moe_normed_buffer,
                        self.rms_norm_eps,
                    )
                    normed = self._moe_normed_buffer
                else:
                    hidden_a.copy_(hidden_b + moe)
                    normed = self._normalization(
                        hidden_a,
                        _layer_tensor(layer + 1, "input_layernorm.weight"),
                    )
            else:
                hidden_a.copy_(hidden_b + moe)
            self._capture_debug(layer, "final_residual_after_mlp", hidden_a)
            if lev is not None:
                lev.mlp_end.record()
                lev.layer_end.record()
            if self.evict_completed_layers and hasattr(
                self.weights, "evict_completed_layer"
            ):
                self.weights.evict_completed_layer(layer)
        return self._normalization(hidden_a, "model.norm.weight")

    def _dense_graph_mlp_forward(
        self,
        layer: int,
        hidden: torch.Tensor,
    ) -> torch.Tensor:
        """Execute the dense feed-forward block bound by the layer graph."""
        plan = self.execution_plan.layers[layer]

        def tensor(role: str) -> torch.Tensor:
            value = self._bound_layer_tensor(layer, role)
            if value is not None:
                return value
            page_id = plan.tensors.get(role)
            if page_id is None:
                raise RuntimeError(
                    f"layer {layer} dense MLP is missing role {role!r}"
                )
            return self.weights.tensor(page_id)

        def bias(role: str) -> Optional[torch.Tensor]:
            value = self._bound_layer_tensor(layer, role)
            if value is not None:
                return value
            page_id = plan.optional_tensors.get(role)
            return self.weights.tensor(page_id) if page_id is not None else None

        if "gate_up_weight" in plan.tensors:
            gate_up_weight = tensor("gate_up_weight")
            gate_up = self._runtime_matvec(
                gate_up_weight,
                hidden,
                torch.empty(
                    int(gate_up_weight.shape[0]),
                    device=hidden.device,
                    dtype=hidden.dtype,
                ),
            )
            gate_up_bias = bias("gate_up_bias")
            if gate_up_bias is not None:
                gate_up.add_(gate_up_bias)
            gate, up = gate_up.chunk(2)
        else:
            gate_weight = tensor("gate_weight")
            up_weight = tensor("up_weight")
            gate = self._runtime_matvec(
                gate_weight,
                hidden,
                torch.empty(
                    int(gate_weight.shape[0]),
                    device=hidden.device,
                    dtype=hidden.dtype,
                ),
            )
            up = self._runtime_matvec(
                up_weight,
                hidden,
                torch.empty(
                    int(up_weight.shape[0]),
                    device=hidden.device,
                    dtype=hidden.dtype,
                ),
            )
            gate_bias = bias("gate_bias")
            up_bias = bias("up_bias")
            if gate_bias is not None:
                gate.add_(gate_bias)
            if up_bias is not None:
                up.add_(up_bias)
        activation = (
            torch.nn.functional.gelu(gate, approximate="tanh") * up
            if self.descriptor.activation == "gelu_pytorch_tanh"
            else torch.nn.functional.silu(gate) * up
        )
        down_weight = tensor("down_weight")
        output = self._runtime_matvec(
            down_weight,
            activation,
            torch.empty(
                int(down_weight.shape[0]),
                device=hidden.device,
                dtype=hidden.dtype,
            ),
        )
        down_bias = bias("down_bias")
        if down_bias is not None:
            output.add_(down_bias)
        return output

    def _packed_moe_forward(
        self,
        layer: int,
        hidden: torch.Tensor,
    ) -> torch.Tensor:
        router_weight = self._bound_layer_tensor(layer, "router_weight")
        if router_weight is None:
            router_weight = _first_optional_tensor(
                self.weights,
                (
                    _layer_tensor(layer, "mlp.router.weight"),
                    _layer_tensor(layer, "block_sparse_moe.gate.weight"),
                    _layer_tensor(layer, "mlp.gate.weight"),
                ),
            )
        if router_weight is None:
            raise RuntimeError(f"layer {layer} has no supported MoE router")
        router_bias = self._bound_layer_tensor(layer, "router_bias")
        if router_bias is None:
            router_bias = _first_optional_tensor(
                self.weights,
                (
                    _layer_tensor(layer, "mlp.router.bias"),
                    _layer_tensor(layer, "block_sparse_moe.gate.bias"),
                ),
            )
        top_k = self.effective_moe_top_k
        if top_k <= 0:
            raise RuntimeError("MoE archive has no num_experts_per_token")
        # Router selection is model work, not benchmark token sampling. Top-k
        # avoids sorting all experts on every MoE layer.
        should_renorm = self.descriptor.norm_topk_prob and (
            self.effective_moe_top_k == self.model_moe_top_k
            or self.moe_top_k_renorm
        )
        if self.kernel_backend is not None and int(router_weight.shape[0]) <= 128:
            router_indices, router_scores = self.kernel_backend.moe_router_matvec_topk(
                router_weight,
                hidden,
                self._moe_router_indices_buffer[:top_k],
                self._moe_router_scores_buffer[:top_k],
                bias=router_bias,
                renormalize=should_renorm,
            )
        elif self.kernel_backend is not None:
            router_logits = torch.mv(
                router_weight,
                hidden,
                out=self._moe_router_logits_buffer,
            )
            if router_bias is not None:
                router_logits.add_(router_bias)
            router_indices, router_scores = self.kernel_backend.moe_router_topk(
                router_logits,
                self._moe_router_indices_buffer[:top_k],
                self._moe_router_scores_buffer[:top_k],
                renormalize=should_renorm,
            )
        else:
            router_logits = torch.mv(router_weight, hidden)
            if router_bias is not None:
                router_logits.add_(router_bias)
            _, router_indices = torch.topk(
                router_logits, k=top_k, dim=0, largest=True, sorted=True
            )
            all_router_scores = torch.softmax(
                router_logits, dim=0, dtype=torch.float32
            )
            router_scores = all_router_scores.index_select(
                0, router_indices
            ).to(dtype=hidden.dtype)
            if should_renorm:
                router_scores = router_scores / router_scores.sum().clamp_min(
                    torch.finfo(router_scores.dtype).tiny
                )

        gate_up = _optional_tensor(
            self.weights,
            _layer_tensor(layer, "mlp.experts.gate_up_proj"),
        )
        packed_input_major = False
        gate_up_output = None
        activation = None
        gate_up_bias = self._bound_layer_tensor(
            layer,
            "experts_gate_up_bias",
        )
        if gate_up_bias is None:
            gate_up_bias = _optional_tensor(
                self.weights,
                _layer_tensor(layer, "mlp.experts.gate_up_proj_bias"),
            )
        if gate_up is not None:
            packed_input_major = (
                int(gate_up.shape[-2]) == int(hidden.numel())
                and int(gate_up.shape[-1])
                == 2 * int(self.intermediate_size)
            )

            def plain_gate_up(expert: torch.Tensor) -> torch.Tensor:
                return (
                    torch.matmul(hidden, expert)
                    if packed_input_major
                    else torch.mv(expert, hidden)
                )

            if top_k == 1:
                idx = int(router_indices[0])
                selected_gate_up = gate_up[idx]
                gate_up_output = plain_gate_up(selected_gate_up)
            else:
                gate_up_outputs = []
                for i in range(top_k):
                    idx = int(router_indices[i])
                    gate_up_outputs.append(plain_gate_up(gate_up[idx]))
                gate_up_output = torch.stack(gate_up_outputs)
        else:
            gate_blocks_id = _layer_tensor(
                layer,
                "mlp.experts.gate_up_proj_blocks",
            )
            gate_scales_id = _layer_tensor(
                layer,
                "mlp.experts.gate_up_proj_scales",
            )
            bound_gate_blocks = self._bound_layer_tensor(
                layer,
                "experts_gate_up",
            )
            bound_gate_scales = self._bound_layer_tensor(
                layer,
                "experts_gate_up_scales",
            )
            if bound_gate_blocks is not None and bound_gate_scales is not None:
                gate_blocks = bound_gate_blocks
                gate_scales = bound_gate_scales
                gate_kernel_indices = router_indices
            elif (
                isinstance(self.weights, ThinGpuPagePool)
                and self.kernel_backend is not None
                and self._moe_gate_up_buffer is not None
                and self.weights.has_page(gate_blocks_id)
                and self.weights.has_page(gate_scales_id)
            ):
                gate_blocks, gate_scales, gate_kernel_indices = (
                    self.weights.mxfp4_selected_expert_tensors(
                        gate_blocks_id,
                        gate_scales_id,
                        router_indices,
                    )
                )
            else:
                gate_blocks = _optional_tensor(self.weights, gate_blocks_id)
                gate_scales = _optional_tensor(self.weights, gate_scales_id)
                gate_kernel_indices = router_indices
            if gate_blocks is None or gate_scales is None:
                return self._separate_expert_moe_forward(
                    layer,
                    hidden,
                    router_indices,
                    router_scores,
                )
            if (
                self.kernel_backend is not None
                and gate_blocks.device.type == "cuda"
                and self._moe_gate_up_buffer is not None
            ):
                gate_bits = (
                    self.weights.packed_expert_bits(gate_blocks_id)
                    if isinstance(self.weights, ThinGpuPagePool)
                    else 4
                )
                if gate_bits < 4 and self.descriptor.swiglu_limit is not None:
                    activation = self.kernel_backend.mxfp_lowbit_selected_swiglu(
                        gate_blocks,
                        gate_scales,
                        hidden,
                        gate_kernel_indices,
                        self._moe_activation_buffer[:top_k],
                        bits=gate_bits,
                        alpha=self.descriptor.swiglu_alpha,
                        limit=self.descriptor.swiglu_limit,
                        bias=gate_up_bias,
                        block_m=8 if gate_bits == 2 else 4,
                        block_k=1024,
                        num_warps=4,
                    )
                    if top_k == 1:
                        activation = activation.squeeze(0)
                    selected_gate_up = None
                else:
                    selected_gate_up = (
                        self.kernel_backend.mxfp_lowbit_selected_matvec(
                            gate_blocks,
                            gate_scales,
                            hidden,
                            gate_kernel_indices,
                            self._moe_gate_up_buffer,
                            bits=gate_bits,
                        )
                        if gate_bits < 4
                        else self.kernel_backend.mxfp4_selected_tensorcore_matvec(
                            gate_blocks,
                            gate_scales,
                            hidden,
                            gate_kernel_indices,
                            self._moe_gate_up_buffer,
                        )
                        if self.mxfp4_selected_tensorcore
                        else self.kernel_backend.mxfp4_selected_matvec(
                            gate_blocks,
                            gate_scales,
                            hidden,
                            gate_kernel_indices,
                            self._moe_gate_up_buffer,
                        )
                    )
                if selected_gate_up is not None:
                    gate_up_output = (
                        selected_gate_up.squeeze(0)
                        if top_k == 1
                        else selected_gate_up
                    )
            else:
                if top_k == 1:
                    idx = int(router_indices[0])
                    selected_gate_up = _dequantize_mxfp4(
                        gate_blocks[idx],
                        gate_scales[idx],
                        dtype=hidden.dtype,
                    )
                    gate_up_output = torch.mv(selected_gate_up, hidden)
                else:
                    gate_up_outputs = []
                    for i in range(top_k):
                        idx = int(router_indices[i])
                        expert_weight = _dequantize_mxfp4(
                            gate_blocks[idx],
                            gate_scales[idx],
                            dtype=hidden.dtype,
                        )
                        gate_up_outputs.append(torch.mv(expert_weight, hidden))
                    gate_up_output = torch.stack(gate_up_outputs)
            del gate_blocks, gate_scales, gate_kernel_indices
        if (
            activation is None
            and gate_up_bias is not None
            and gate_up_output is not None
        ):
            if top_k == 1:
                idx = int(router_indices[0])
                gate_up_output.add_(gate_up_bias[idx])
            else:
                for i in range(top_k):
                    idx = int(router_indices[i])
                    gate_up_output[i].add_(gate_up_bias[idx])
        if activation is None and self.descriptor.swiglu_limit is not None:
            if self.kernel_backend is not None:
                activation = self.kernel_backend.interleaved_swiglu(
                    gate_up_output.reshape(top_k, -1),
                    self._moe_activation_buffer[:top_k],
                    alpha=self.descriptor.swiglu_alpha,
                    limit=self.descriptor.swiglu_limit,
                )
                if top_k == 1:
                    activation = activation.squeeze(0)
            else:
                if top_k == 1:
                    gate = gate_up_output[0::2].clamp(
                        max=self.descriptor.swiglu_limit
                    )
                    up = gate_up_output[1::2].clamp(
                        min=-self.descriptor.swiglu_limit,
                        max=self.descriptor.swiglu_limit,
                    )
                else:
                    gate = gate_up_output[:, 0::2].clamp(
                        max=self.descriptor.swiglu_limit
                    )
                    up = gate_up_output[:, 1::2].clamp(
                        min=-self.descriptor.swiglu_limit,
                        max=self.descriptor.swiglu_limit,
                    )
                activation = (
                    (up + 1)
                    * gate
                    * torch.sigmoid(gate * self.descriptor.swiglu_alpha)
                )
        elif activation is None:
            gate, up = gate_up_output.chunk(2, dim=-1)
            activation = torch.nn.functional.silu(gate) * up

        down = _optional_tensor(
            self.weights,
            _layer_tensor(layer, "mlp.experts.down_proj"),
        )
        expert_output = None
        combined_expert_output = None
        down_bias = self._bound_layer_tensor(layer, "experts_down_bias")
        if down_bias is None:
            down_bias = _optional_tensor(
                self.weights,
                _layer_tensor(layer, "mlp.experts.down_proj_bias"),
            )
        if down is not None:
            def plain_down(
                expert: torch.Tensor,
                expert_activation: torch.Tensor,
            ) -> torch.Tensor:
                return (
                    torch.matmul(expert_activation, expert)
                    if packed_input_major
                    else torch.mv(expert, expert_activation)
                )

            if top_k == 1:
                idx = int(router_indices[0])
                selected_down = down[idx]
                expert_output = plain_down(selected_down, activation)
            else:
                down_outputs = []
                for i in range(top_k):
                    idx = int(router_indices[i])
                    down_outputs.append(
                        plain_down(down[idx], activation[i])
                    )
                expert_output = torch.stack(down_outputs)
        else:
            down_blocks_id = _layer_tensor(
                layer,
                "mlp.experts.down_proj_blocks",
            )
            down_scales_id = _layer_tensor(
                layer,
                "mlp.experts.down_proj_scales",
            )
            bound_down_blocks = self._bound_layer_tensor(
                layer,
                "experts_down",
            )
            bound_down_scales = self._bound_layer_tensor(
                layer,
                "experts_down_scales",
            )
            if bound_down_blocks is not None and bound_down_scales is not None:
                down_blocks = bound_down_blocks
                down_scales = bound_down_scales
                down_kernel_indices = router_indices
            elif (
                isinstance(self.weights, ThinGpuPagePool)
                and self.kernel_backend is not None
                and self._moe_down_buffer is not None
                and self.weights.has_page(down_blocks_id)
                and self.weights.has_page(down_scales_id)
            ):
                down_blocks, down_scales, down_kernel_indices = (
                    self.weights.mxfp4_selected_expert_tensors(
                        down_blocks_id,
                        down_scales_id,
                        router_indices,
                    )
                )
            else:
                down_blocks = _optional_tensor(self.weights, down_blocks_id)
                down_scales = _optional_tensor(self.weights, down_scales_id)
                down_kernel_indices = router_indices
            if down_blocks is None or down_scales is None:
                raise RuntimeError(
                    f"layer {layer} has no supported packed expert down tensor"
                )
            if (
                self.kernel_backend is not None
                and down_blocks.device.type == "cuda"
                and self._moe_down_buffer is not None
            ):
                down_bits = (
                    self.weights.packed_expert_bits(down_blocks_id)
                    if isinstance(self.weights, ThinGpuPagePool)
                    else 4
                )
                if down_bits < 4 and top_k > 1:
                    combined_expert_output = (
                        self.kernel_backend.mxfp_lowbit_weighted_down(
                            down_blocks,
                            down_scales,
                            activation,
                            down_kernel_indices,
                            router_scores,
                            self._moe_combined_buffer,
                            bits=down_bits,
                            bias=down_bias,
                            block_m=8,
                            block_k=1024 if down_bits == 2 else 512,
                            num_warps=4,
                        )
                    )
                    expert_output = None
                else:
                    expert_output = (
                        self.kernel_backend.mxfp_lowbit_selected_matvec(
                            down_blocks,
                            down_scales,
                            activation,
                            down_kernel_indices,
                            self._moe_down_buffer,
                            bits=down_bits,
                        )
                        if down_bits < 4
                        else self.kernel_backend.mxfp4_selected_tensorcore_matvec(
                            down_blocks,
                            down_scales,
                            activation,
                            down_kernel_indices,
                            self._moe_down_buffer,
                        )
                        if self.mxfp4_selected_tensorcore
                        else self.kernel_backend.mxfp4_selected_matvec(
                            down_blocks,
                            down_scales,
                            activation,
                            down_kernel_indices,
                            self._moe_down_buffer,
                        )
                    )
                if top_k == 1 and expert_output is not None:
                    expert_output = expert_output.squeeze(0)
            else:
                if top_k == 1:
                    idx = int(router_indices[0])
                    selected_down = _dequantize_mxfp4(
                        down_blocks[idx],
                        down_scales[idx],
                        dtype=hidden.dtype,
                    )
                    expert_output = torch.mv(selected_down, activation)
                else:
                    down_outputs = []
                    for i in range(top_k):
                        idx = int(router_indices[i])
                        expert_weight = _dequantize_mxfp4(
                            down_blocks[idx],
                            down_scales[idx],
                            dtype=hidden.dtype,
                        )
                        down_outputs.append(torch.mv(expert_weight, activation[i]))
                    expert_output = torch.stack(down_outputs)
            del down_blocks, down_scales, down_kernel_indices
        if combined_expert_output is not None:
            return combined_expert_output
        if down_bias is not None and expert_output is not None:
            if top_k == 1:
                idx = int(router_indices[0])
                expert_output.add_(down_bias[idx])
            else:
                for i in range(top_k):
                    idx = int(router_indices[i])
                    expert_output[i].add_(down_bias[idx])
        if top_k == 1:
            return (expert_output * router_scores[0]).to(dtype=hidden.dtype)
        elif self.kernel_backend is not None:
            return self.kernel_backend.weighted_expert_sum(
                expert_output,
                router_scores,
                self._moe_combined_buffer,
            )
        else:
            return torch.sum(
                expert_output * router_scores[:, None],
                dim=0,
            ).to(dtype=hidden.dtype)

    def _separate_expert_moe_forward(
        self,
        layer: int,
        hidden: torch.Tensor,
        router_indices: torch.Tensor,
        router_scores: torch.Tensor,
    ) -> torch.Tensor:
        """Execute only the experts selected by the exact GPU router."""
        if self._moe_gate_up_buffer is None or self._moe_down_buffer is None:
            raise RuntimeError("MoE expert buffers were not initialized")
        if (
            isinstance(self.weights, ThinGpuPagePool)
            and self.weights.has_expert_int4_pack(layer)
        ):
            if self.kernel_backend is None:
                raise RuntimeError(
                    "packed expert INT4 requires a Triton backend"
                )
            pack_ids = self.weights._expert_int4_pack_page_ids[layer]
            self.weights._active_pages.update(pack_ids)
            try:
                gate_up_q, gate_up_s, down_q, down_s = (
                    self.weights.expert_int4_pack(layer)
                )
                gate_up = self._moe_gate_up_buffer[
                    : int(router_indices.numel())
                ]
                self.kernel_backend.int4_selected_matvec(
                    gate_up_q,
                    gate_up_s,
                    hidden,
                    router_indices,
                    gate_up,
                    group_size=self.weights.expert_int4_group_size,
                )
                gate, up = gate_up.chunk(2, dim=-1)
                activation = torch.nn.functional.silu(gate) * up
                expert_output = self._moe_down_buffer[
                    : int(router_indices.numel())
                ]
                self.kernel_backend.int4_selected_matvec(
                    down_q,
                    down_s,
                    activation,
                    router_indices,
                    expert_output,
                    group_size=self.weights.expert_int4_group_size,
                )
                return torch.sum(
                    expert_output * router_scores[:, None],
                    dim=0,
                ).to(dtype=hidden.dtype)
            finally:
                self.weights._active_pages.difference_update(pack_ids)
        # Selected expert IDs are the only dynamic page addresses. This one
        # small synchronization prevents loading all experts in the layer.
        selected = [int(value) for value in router_indices.tolist()]
        gate_up = self._moe_gate_up_buffer[: len(selected)]
        expert_output = self._moe_down_buffer[: len(selected)]
        intermediate = self.intermediate_size
        for slot, expert in enumerate(selected):
            gate_id, up_id, down_id = _separate_expert_tensor_ids(
                self.weights,
                layer,
                expert,
            )
            self._expert_page_matvec(
                gate_id,
                hidden,
                gate_up[slot, :intermediate],
            )
            self._expert_page_matvec(
                up_id,
                hidden,
                gate_up[slot, intermediate:],
            )
            activation = (
                torch.nn.functional.silu(gate_up[slot, :intermediate])
                * gate_up[slot, intermediate:]
            )
            self._expert_page_matvec(
                down_id,
                activation,
                expert_output[slot],
            )
        combined = torch.zeros_like(hidden)
        for slot in sorted(range(len(selected)), key=selected.__getitem__):
            combined.add_(expert_output[slot] * router_scores[slot])
        return combined

    def _expert_page_matvec(
        self,
        page_id: str,
        x: torch.Tensor,
        out: torch.Tensor,
    ) -> torch.Tensor:
        weight = self.weights.tensor(page_id)
        if isinstance(self.weights, ThinGpuPagePool):
            metadata = self.weights._int4_metadata_by_page.get(page_id)
            if metadata is not None:
                if self.kernel_backend is None:
                    raise RuntimeError(
                        "streamed expert INT4 requires a Triton backend"
                    )
                rows, cols, group_size = metadata
                # Retain both CUDA tensors locally through dispatch. Loading
                # the scale may evict the weight from the page-pool index under
                # an extremely small budget, but record_stream keeps storage
                # alive until this kernel finishes.
                self.weights._active_pages.add(page_id)
                try:
                    scale_id = page_id + ".scale"
                    scale = self.weights.tensor(scale_id)
                    result = self.kernel_backend.int4_scaled_matvec(
                        weight,
                        scale,
                        x,
                        out,
                        rows=rows,
                        cols=cols,
                        group_size=group_size,
                    )
                    event = torch.cuda.Event()
                    event.record(torch.cuda.current_stream(weight.device))
                    self.weights._page_use_events[page_id] = event
                    self.weights._page_use_events[scale_id] = event
                    return result
                finally:
                    self.weights._active_pages.discard(page_id)
            # Exact expert pages are short-lived and may be reloaded at a new
            # address after LRU eviction. cuBLAS matvec has stable lifetime
            # semantics for this path; the hand-tuned Triton dispatch is kept
            # for persistent dense pages and packed INT4 experts.
            torch.mv(weight, x, out=out)
            return out
        return self._runtime_matvec(weight, x, out)

    @torch.inference_mode()
    def _forward_token_pytorch_fallback(
        self,
        embed: torch.Tensor,
        token_id: int | torch.Tensor,
        layer_count: int,
        token_index: int,
    ) -> torch.Tensor:
        buffers = (
            self.kernel_backend.buffers
            if self.kernel_backend is not None
            else None
        )
        if buffers is not None:
            hidden = self._copy_embed_token(
                embed, token_id, buffers.hidden_a
            )
        elif isinstance(self.weights, ThinGpuPagePool) and self.weights.cpu_embed:
            hidden = torch.empty(
                self.hidden_size,
                device=self.device,
                dtype=self.dtype,
            )
            hidden = self._copy_embed_token(embed, token_id, hidden)
        else:
            hidden = embed[token_id].clone()
        if self.embedding_scale != 1.0:
            hidden.mul_(
                torch.tensor(
                    self.embedding_scale,
                    device=hidden.device,
                    dtype=hidden.dtype,
                )
            )
        if not hasattr(self, "_fallback_native_matvec_enabled"):
            free_bytes = (
                torch.cuda.mem_get_info(self.device)[0]
                if self.device.type == "cuda"
                else 0
            )
            # Triton may need driver/compiler workspace on its first launch.
            # A nearly full BF16 model must retain the already-working cuBLAS
            # path rather than failing while trying to JIT a faster kernel.
            self._fallback_native_matvec_enabled = (
                self.kernel_backend is not None
                and free_bytes >= 128 * 1024 * 1024
            )
        native_matvec = bool(self._fallback_native_matvec_enabled)

        def projection(
            weight: torch.Tensor,
            value: torch.Tensor,
            out: torch.Tensor | None,
        ) -> torch.Tensor:
            requires_native = weight.dtype in {
                torch.float8_e4m3fn,
                torch.int8,
                torch.uint8,
            }
            if (
                out is not None
                and self.kernel_backend is not None
                and (native_matvec or requires_native)
            ):
                return self._runtime_matvec(weight, value, out)
            return torch.mv(weight, value)

        for layer in range(layer_count):
            self._capture_debug(layer, "layer_input", hidden)
            if getattr(self, "_profiler_enabled", False):
                lev = LayerProfileEvents()
                self._current_step_profile["layers"].append(lev)
                lev.layer_start.record()
                lev.attn_start.record()
                lev.qkv_start.record()
            for distance in range(1, self.prefetch_distance + 1):
                if hasattr(self.weights, "prefetch_layer"):
                    self.weights.prefetch_layer(layer + distance)
            normed = self._normalization(
                hidden,
                _layer_tensor(layer, "input_layernorm.weight"),
                out=buffers.normed if buffers is not None else None,
            )
            self._capture_debug(layer, "post_input_rmsnorm", normed)
            qkv = self._fused_qkv(layer, normed)
            if qkv is None:
                fused_qkv_weight = _optional_tensor(
                    self.weights,
                    _layer_tensor(
                        layer,
                        "self_attn.qkv_proj.weight",
                    ),
                )
                if fused_qkv_weight is not None:
                    projected = projection(
                        fused_qkv_weight,
                        normed,
                        buffers.qkv if buffers is not None else None,
                    )
                    q, k, v = torch.split(
                        projected,
                        (self.q_dim, self.kv_dim, self.kv_dim),
                    )
                else:
                    q = projection(
                        self.weights.tensor(_layer_tensor(layer, "self_attn.q_proj.weight")),
                        normed,
                        buffers.qkv[: self.q_dim] if buffers is not None else None,
                    )
                    k = projection(
                        self.weights.tensor(_layer_tensor(layer, "self_attn.k_proj.weight")),
                        normed,
                        (
                            buffers.qkv[self.q_dim : self.q_dim + self.kv_dim]
                            if buffers is not None
                            else None
                        ),
                    )
                    v = projection(
                        self.weights.tensor(_layer_tensor(layer, "self_attn.v_proj.weight")),
                        normed,
                        (
                            buffers.qkv[
                                self.q_dim + self.kv_dim :
                                self.q_dim + 2 * self.kv_dim
                            ]
                            if buffers is not None
                            else None
                        ),
                    )
            else:
                q, k, v = qkv
            for projected, suffix in (
                (q, "self_attn.q_proj.bias"),
                (k, "self_attn.k_proj.bias"),
                (v, "self_attn.v_proj.bias"),
            ):
                bias = _optional_tensor(
                    self.weights, _layer_tensor(layer, suffix)
                )
                if bias is not None:
                    projected.add_(bias)
            self._capture_debug(layer, "q_projection", q)
            self._capture_debug(layer, "k_projection", k)
            self._capture_debug(layer, "v_projection", v)
            if getattr(self, "_profiler_enabled", False):
                lev.qkv_end.record()
            q_norm = _optional_tensor(self.weights, _layer_tensor(layer, "self_attn.q_norm.weight"))
            k_norm = _optional_tensor(self.weights, _layer_tensor(layer, "self_attn.k_norm.weight"))
            if q_norm is not None:
                q = _head_rms_norm(q, q_norm, self.heads, self.head_dim, self.rms_norm_eps)
            if k_norm is not None:
                k = _head_rms_norm(k, k_norm, self.kv_heads, self.head_dim, self.rms_norm_eps)
            self._capture_debug(layer, "q_after_q_norm", q)
            self._capture_debug(layer, "k_after_k_norm", k)
            if self.attention_mode == "causal_kv":
                q, k = self._apply_rope(q, k, layer, token_index)
            self._capture_debug(layer, "q_after_rope", q)
            self._capture_debug(layer, "k_after_rope", k)
            if self.kv_cache is not None and not getattr(
                self, "_autotuning", False
            ):
                self.kv_cache.append(layer, k, v, token_index)

            attn = self._attention(
                q,
                k,
                v,
                layer,
                token_index,
                lev if getattr(self, "_profiler_enabled", False) else None,
            )
            if getattr(self, "_profiler_enabled", False):
                lev.o_proj_start.record()
            attn_out = projection(
                self.weights.tensor(_layer_tensor(layer, "self_attn.o_proj.weight")),
                attn,
                buffers.attn_out if buffers is not None else None,
            )
            o_bias = _optional_tensor(
                self.weights,
                _layer_tensor(layer, "self_attn.o_proj.bias"),
            )
            if o_bias is not None:
                attn_out.add_(o_bias)
            self._capture_debug(layer, "o_proj_output", attn_out)
            if getattr(self, "_profiler_enabled", False):
                lev.o_proj_end.record()
            if self.uses_dual_post_norm:
                attn_out = self._normalization(
                    attn_out,
                    _layer_tensor(
                        layer,
                        "post_attention_layernorm.weight",
                    ),
                    out=buffers.normed if buffers is not None else None,
                )
                self._capture_debug(
                    layer,
                    "post_attention_rmsnorm",
                    attn_out,
                )
            hidden = hidden + attn_out
            self._capture_debug(layer, "post_attention_residual", hidden)
            if getattr(self, "_profiler_enabled", False):
                lev.attn_end.record()
                lev.mlp_start.record()

            mlp_norm_suffix = (
                "pre_feedforward_layernorm.weight"
                if self.uses_dual_post_norm
                else "post_attention_layernorm.weight"
            )
            normed = self._normalization(
                hidden,
                _layer_tensor(layer, mlp_norm_suffix),
                out=buffers.normed if buffers is not None else None,
            )
            if self.uses_dual_post_norm:
                self._capture_debug(
                    layer,
                    "pre_feedforward_rmsnorm",
                    normed,
                )
            else:
                self._capture_debug(
                    layer,
                    "post_attention_rmsnorm",
                    normed,
                )
            if getattr(self, "_profiler_enabled", False):
                lev.gate_proj_start.record()
            gate_up = self._fused_gate_up(layer, normed)
            if gate_up is None:
                fused_gate_up = _optional_tensor(
                    self.weights,
                    _layer_tensor(layer, "mlp.gate_up_proj.weight"),
                )
                if fused_gate_up is not None:
                    gate, up = projection(
                        fused_gate_up,
                        normed,
                        buffers.gate_up if buffers is not None else None,
                    ).chunk(2)
                else:
                    gate = projection(
                        self.weights.tensor(_layer_tensor(layer, "mlp.gate_proj.weight")),
                        normed,
                        (
                            buffers.gate_up[: self.intermediate_size]
                            if buffers is not None
                            else None
                        ),
                    )
                    up = projection(
                        self.weights.tensor(_layer_tensor(layer, "mlp.up_proj.weight")),
                        normed,
                        (
                            buffers.gate_up[self.intermediate_size :]
                            if buffers is not None
                            else None
                        ),
                    )
            else:
                gate, up = gate_up
            gate_bias = _optional_tensor(
                self.weights, _layer_tensor(layer, "mlp.gate_proj.bias")
            )
            up_bias = _optional_tensor(
                self.weights, _layer_tensor(layer, "mlp.up_proj.bias")
            )
            if gate_bias is not None:
                gate.add_(gate_bias)
            if up_bias is not None:
                up.add_(up_bias)
            fused_gate_up_bias = _optional_tensor(
                self.weights,
                _layer_tensor(layer, "mlp.gate_up_proj.bias"),
            )
            if fused_gate_up_bias is not None:
                gate_bias, up_bias = fused_gate_up_bias.chunk(2)
                gate.add_(gate_bias)
                up.add_(up_bias)
            self._capture_debug(layer, "gate_projection", gate)
            self._capture_debug(layer, "up_projection", up)
            if getattr(self, "_profiler_enabled", False):
                lev.gate_proj_end.record()
                lev.silu_mul_start.record()
            if self.descriptor.activation == "gelu_pytorch_tanh":
                silu_val = torch.nn.functional.gelu(
                    gate,
                    approximate="tanh",
                )
                silu_val.mul_(up)
                if buffers is not None:
                    buffers.mlp_act.copy_(silu_val)
                    silu_val = buffers.mlp_act
            elif buffers is not None and self.kernel_backend is not None:
                silu_val = self._runtime_silu_mul(
                    gate,
                    up,
                    buffers.mlp_act,
                )
            else:
                silu_val = torch.nn.functional.silu(gate) * up
            self._capture_debug(layer, "gated_activation", silu_val)
            if getattr(self, "_profiler_enabled", False):
                lev.silu_mul_end.record()
                lev.down_proj_start.record()
            mlp = projection(
                self.weights.tensor(_layer_tensor(layer, "mlp.down_proj.weight")),
                silu_val,
                buffers.mlp if buffers is not None else None,
            )
            down_bias = _optional_tensor(
                self.weights,
                _layer_tensor(layer, "mlp.down_proj.bias"),
            )
            if down_bias is not None:
                mlp.add_(down_bias)
            self._capture_debug(layer, "down_projection", mlp)
            if self.uses_dual_post_norm:
                mlp = self._normalization(
                    mlp,
                    _layer_tensor(
                        layer,
                        "post_feedforward_layernorm.weight",
                    ),
                    out=buffers.normed if buffers is not None else None,
                )
                self._capture_debug(
                    layer,
                    "post_feedforward_rmsnorm",
                    mlp,
                )
            if getattr(self, "_profiler_enabled", False):
                lev.down_proj_end.record()
            hidden = hidden + mlp
            self._capture_debug(layer, "final_residual_after_mlp", hidden)
            if self.evict_completed_layers and hasattr(self.weights, "evict_completed_layer"):
                self.weights.evict_completed_layer(layer)
            if getattr(self, "_profiler_enabled", False):
                lev.mlp_end.record()
                lev.layer_end.record()

        hidden = self._normalization(hidden, "model.norm.weight")
        return hidden

    def _lm_head(self) -> torch.Tensor:
        if self._lm_head_tensor is None:
            head = _optional_tensor(self.weights, "lm_head.weight")
            if head is None:
                head = self.weights.tensor("model.embed_tokens.weight")
            self._lm_head_tensor = head
        return self._lm_head_tensor

    def _make_layer_plan(self, layer: int) -> LayerPlan:
        q_id = _layer_tensor(layer, "self_attn.q_proj.weight")
        k_id = _layer_tensor(layer, "self_attn.k_proj.weight")
        v_id = _layer_tensor(layer, "self_attn.v_proj.weight")
        gate_id = _layer_tensor(layer, "mlp.gate_proj.weight")
        up_id = _layer_tensor(layer, "mlp.up_proj.weight")
        return LayerPlan(
            layer=layer,
            input_layernorm_weight=self.weights.tensor(
                _layer_tensor(layer, "input_layernorm.weight")
            ),
            q_proj=self.weights.tensor(q_id),
            k_proj=self.weights.tensor(k_id),
            v_proj=self.weights.tensor(v_id),
            o_proj=self.weights.tensor(_layer_tensor(layer, "self_attn.o_proj.weight")),
            q_norm=_optional_tensor(
                self.weights, _layer_tensor(layer, "self_attn.q_norm.weight")
            ),
            k_norm=_optional_tensor(
                self.weights, _layer_tensor(layer, "self_attn.k_norm.weight")
            ),
            post_attention_layernorm_weight=self.weights.tensor(
                _layer_tensor(layer, "post_attention_layernorm.weight")
            ),
            gate_proj=self.weights.tensor(gate_id),
            up_proj=self.weights.tensor(up_id),
            down_proj=self.weights.tensor(_layer_tensor(layer, "mlp.down_proj.weight")),
            qkv_fused=self._fused_matrix(
                self.weights,
                f"layer_{layer}_attn_qkv_fused",
                q_id,
                [q_id, k_id, v_id],
            ) if not self.qkv_fp8 else None,
            gate_up_fused=self._fused_matrix(
                self.weights,
                f"layer_{layer}_mlp_gate_up_fused",
                gate_id,
                [gate_id, up_id],
            ) if not self.gate_up_fp8 else None,
        )

    def _initialize_runtime_fusion(self) -> None:
        if not self.runtime_fusion_enabled or self._layer_plan is None:
            return
        for plan in self._layer_plan:
            if plan.qkv_fused is None:
                plan.qkv_fused = torch.cat(
                    [plan.q_proj, plan.k_proj, plan.v_proj], dim=0
                ).contiguous()
                self._runtime_fused_qkv_cache[plan.layer] = plan.qkv_fused
                self.runtime_fusion_extra_bytes += _tensor_nbytes(plan.qkv_fused)
            if plan.gate_up_fused is None:
                plan.gate_up_fused = torch.cat(
                    [plan.gate_proj, plan.up_proj], dim=0
                ).contiguous()
                self._runtime_fused_gate_up_cache[plan.layer] = plan.gate_up_fused
                self.runtime_fusion_extra_bytes += _tensor_nbytes(plan.gate_up_fused)

    @staticmethod
    def _matvec_shape_key(weight: torch.Tensor) -> tuple[Any, ...]:
        return (
            int(weight.shape[0]),
            int(weight.shape[1]),
            str(weight.dtype),
            int(weight.stride(0)),
            int(weight.stride(1)),
        )

    def _initialize_matvec_backend_choices(self) -> None:
        backend = self.kernel_backend
        if backend is None:
            return
        from .tuning_cache import (
            TuningCache,
            builtin_matvec_choice,
            matvec_key,
        )

        capability = torch.cuda.get_device_capability(self.device)
        gpu_name = torch.cuda.get_device_name(self.device)
        tuning_cache = TuningCache(
            f"{gpu_name}-cc{capability[0]}{capability[1]}-"
            f"torch{torch.__version__}"
        )
        candidates: list[torch.Tensor] = []
        if isinstance(self.weights, ThinGpuPagePool) and self._layer_plan is not None:
            # Streaming tensors are reloadable and dispatch is keyed by stable
            # shape/dtype/stride. Load one representative of each shape rather
            # than materializing every layer merely to build the backend table.
            representative_keys: set[tuple[Any, ...]] = set()
            candidate_ids: list[str] = []
            for plan in self._layer_plan:
                candidate_ids.extend(
                    [
                        plan.qkv_fused_id
                        if self.weights.has_page(plan.qkv_fused_id)
                        else plan.q_proj_id,
                        plan.k_proj_id,
                        plan.v_proj_id,
                        plan.o_proj_id,
                        plan.gate_up_fused_id
                        if self.weights.has_page(plan.gate_up_fused_id)
                        else plan.gate_proj_id,
                        plan.up_proj_id,
                        plan.down_proj_id,
                    ]
                )
            for page_id in candidate_ids:
                spec = self.weights.page_specs.get(page_id)
                if spec is None:
                    continue
                source_dtype = map_dtype(spec["dtype"])
                target_dtype = _target_dtype(source_dtype, self.weights.dtype)
                shape = tuple(int(dim) for dim in spec["shape"])
                if len(shape) != 2:
                    continue
                key = (shape[0], shape[1], str(target_dtype), shape[1], 1)
                if key in representative_keys:
                    continue
                representative_keys.add(key)
                candidates.append(self.weights.tensor(page_id))
        elif self._layer_plan is not None:
            for plan in self._layer_plan:
                candidates.extend(
                    [
                        plan.qkv_fused if plan.qkv_fused is not None else plan.q_proj,
                        plan.k_proj,
                        plan.v_proj,
                        plan.o_proj,
                        plan.gate_up_fused
                        if plan.gate_up_fused is not None
                        else plan.gate_proj,
                        plan.up_proj,
                        plan.down_proj,
                    ]
                )
        elif isinstance(self.weights, ThinGpuWeights):
            # Architectures such as Phi store QKV and gate/up as fused pages.
            # They still use the same shape-driven backend selection.
            for layer in range(self.layers):
                page_ids = [
                    _layer_tensor(layer, "self_attn.o_proj.weight"),
                    _layer_tensor(layer, "mlp.down_proj.weight"),
                ]
                page_ids.extend(
                    [
                        _layer_tensor(layer, "self_attn.qkv_proj.weight"),
                    ]
                    if self.has_fused_qkv_projection
                    else [
                        _layer_tensor(layer, f"self_attn.{part}_proj.weight")
                        for part in ("q", "k", "v")
                    ]
                )
                page_ids.extend(
                    [
                        _layer_tensor(layer, "mlp.gate_up_proj.weight"),
                    ]
                    if self.has_fused_gate_up_projection
                    else [
                        _layer_tensor(layer, f"mlp.{part}_proj.weight")
                        for part in ("gate", "up")
                    ]
                )
                for page_id in page_ids:
                    if page_id in self.weights.tensors:
                        candidates.append(self.weights.tensor(page_id))
        head = self._lm_head()
        if not isinstance(self.weights, ThinGpuPagePool) or self._matvec_shape_key(
            head
        ) not in {self._matvec_shape_key(weight) for weight in candidates}:
            candidates.append(head)

        unique: dict[tuple[Any, ...], torch.Tensor] = {}
        for weight in candidates:
            if id(weight) in self._int4_metadata:
                self._matvec_choice_by_tensor_id[id(weight)] = "int4"
                continue
            unique.setdefault(self._matvec_shape_key(weight), weight)

        for key, weight in unique.items():
            rows, cols = int(weight.shape[0]), int(weight.shape[1])
            dtype_str = str(weight.dtype)
            scale = self._weight_scale(weight)
            persistent_key = matvec_key(
                capability=capability,
                rows=rows,
                cols=cols,
                dtype=dtype_str,
                scaled=scale is not None,
                stride=(int(weight.stride(0)), int(weight.stride(1))),
            )
            cached_choice = tuning_cache.get_matvec(persistent_key)
            seeded_choice = builtin_matvec_choice(
                capability,
                rows,
                cols,
                dtype_str,
                scale is not None,
            )
            established_choice = cached_choice or seeded_choice
            if established_choice is not None:
                self._matvec_backend_choices[key] = established_choice
                shape_str = f"{rows}x{cols}:{weight.dtype}:stride={weight.stride(0)},{weight.stride(1)}"
                self.per_shape_backend_benchmarks[shape_str] = {
                    "chosen_backend": established_choice,
                    "winner_microbench": established_choice,
                    "winner_real_decode": established_choice,
                    "source": (
                        "persistent_gpu_cache"
                        if cached_choice is not None
                        else "builtin_gpu_scoped_decode_evidence"
                    ),
                }
                continue

            if not self.autotune_enabled:
                self._matvec_backend_choices[key] = "triton"
                continue
            activation_dtype = (
                self.kernel_backend.dtype if scale is not None else weight.dtype
            )
            x = torch.empty(cols, device=self.device, dtype=activation_dtype)
            out = torch.empty(rows, device=self.device, dtype=activation_dtype)

            choices = (
                ["triton"]
                if scale is not None
                else [
                    "torch_mv",
                    "torch_matmul",
                    "triton",
                    "row_block_m2",
                    "row_block_m4",
                    "row_block_m8",
                ]
            )
            if cols >= 2048:
                choices.extend([
                    "triton_loop_64",
                    "triton_loop_128",
                    "triton_loop_256",
                    "triton_loop_512"
                ])

            def run_choice(choice: str) -> None:
                if scale is not None:
                    backend.scaled_matvec(
                        weight,
                        scale,
                        x,
                        out,
                        config_name=(
                            choice if choice.startswith("triton_loop_") else None
                        ),
                    )
                elif choice == "torch_mv":
                    torch.mv(weight, x, out=out)
                elif choice == "torch_matmul":
                    torch.matmul(weight, x, out=out)
                elif choice == "triton":
                    backend.matvec(weight, x, out)
                elif choice.startswith("triton_loop_"):
                    backend.matvec(weight, x, out, config_name=choice)
                elif choice.startswith("row_block_m"):
                    backend.matvec(weight, x, out, config_name=choice)

            def elapsed_ms(choice: str) -> float:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(20):
                    run_choice(choice)
                end.record()
                end.synchronize()
                return start.elapsed_time(end)

            results = {}
            for choice in choices:
                try:
                    for _ in range(5):
                        run_choice(choice)
                    
                    samples = []
                    for _ in range(3):
                        samples.append(elapsed_ms(choice))
                    results[choice] = statistics.median(samples)
                except Exception:
                    continue

            winner_microbench = "triton"
            if results:
                winner_microbench = min(results, key=results.get)

            # Real decode validation
            decode_results = {}
            for choice in results.keys():
                # Temporarily map all matching tensors to this choice
                for w in candidates:
                    if self._matvec_shape_key(w) == key:
                        self._matvec_choice_by_tensor_id[id(w)] = choice
                
                try:
                    if self.kv_cache is not None:
                        self.kv_cache.reset(reuse_pages=True)
                    token0 = torch.zeros((), device=self.device, dtype=torch.long)
                    for warmup_index in range(3):
                        self.forward_token(
                            token0,
                            token_index=warmup_index,
                        )
                    torch.cuda.synchronize()
                    if self.kv_cache is not None:
                        self.kv_cache.reset(reuse_pages=True)

                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    for step_idx in range(15):
                        self.forward_token(token0, token_index=step_idx)
                    end.record()
                    end.synchronize()
                    decode_results[choice] = start.elapsed_time(end)
                except Exception:
                    continue

            winner_real_decode = winner_microbench
            if decode_results:
                winner_real_decode = min(decode_results, key=decode_results.get)

            self._matvec_backend_choices[key] = winner_real_decode
            tuning_cache.put_matvec(
                persistent_key,
                choice=winner_real_decode,
                microbench_ms=results,
                decode_ms=decode_results,
            )

            # Task 4: Store telemetry
            shape_str = f"{rows}x{cols}:{weight.dtype}:stride={weight.stride(0)},{weight.stride(1)}"
            self.per_shape_backend_benchmarks[shape_str] = {
                "torch_mv_ms": results.get("torch_mv"),
                "torch_matmul_ms": results.get("torch_matmul"),
                "triton_basic_ms": results.get("triton"),
                "triton_loop_64_ms": results.get("triton_loop_64"),
                "triton_loop_128_ms": results.get("triton_loop_128"),
                "triton_loop_256_ms": results.get("triton_loop_256"),
                "triton_loop_512_ms": results.get("triton_loop_512"),
                "row_block_m2_ms": results.get("row_block_m2"),
                "row_block_m4_ms": results.get("row_block_m4"),
                "row_block_m8_ms": results.get("row_block_m8"),
                "winner_microbench": winner_microbench,
                "winner_real_decode": winner_real_decode,
                "chosen_backend": winner_real_decode,
                "source": "measured_and_persisted",
            }


        # Populate Choice by Tensor ID for all layer weights
        for weight in candidates:
            if id(weight) in self._int4_metadata:
                self._matvec_choice_by_tensor_id[id(weight)] = "int4"
                continue
            key = self._matvec_shape_key(weight)
            choice = self._matvec_backend_choices.get(key, "triton")
            self._matvec_choice_by_tensor_id[id(weight)] = choice

    @property
    def per_shape_backend_choices(self) -> dict[str, str]:
        return {
            f"{rows}x{cols}:{dtype}:stride={stride0},{stride1}": choice
            for (rows, cols, dtype, stride0, stride1), choice in self._matvec_backend_choices.items()
        }

    @property
    def per_shape_backend_benchmarks_choices(self) -> dict:
        return self.per_shape_backend_benchmarks

    @property
    def launches_per_token(self) -> int:
        return self.matvec_launches_per_token + self.elementwise_launches_per_token + self.attention_launches_per_token

    @property
    def matvec_launches_per_token(self) -> int:
        has_fused = False
        if self._layer_plan is not None:
            if isinstance(self.weights, ThinGpuPagePool):
                has_fused = self.weights.has_page(self._layer_plan[0].qkv_fused_id)
            else:
                has_fused = self._layer_plan[0].qkv_fused is not None
        qkv_launches = 1 if has_fused or self.use_triton_matvec else 3
        gate_up_launches = 1
        o_proj_launches = 1
        down_proj_launches = 1
        lm_head_launches = 1
        return self.layers * (qkv_launches + o_proj_launches + gate_up_launches + down_proj_launches) + lm_head_launches

    @property
    def elementwise_launches_per_token(self) -> int:
        q_norm_present = 0
        k_norm_present = 0
        if self._layer_plan is not None:
            if isinstance(self.weights, ThinGpuPagePool):
                q_norm_present = 1 if self.weights.has_page(self._layer_plan[0].q_norm_id) else 0
                k_norm_present = 1 if self.weights.has_page(self._layer_plan[0].k_norm_id) else 0
            else:
                q_norm_present = 1 if self._layer_plan[0].q_norm is not None else 0
                k_norm_present = 1 if self._layer_plan[0].k_norm is not None else 0
        silu_mul_launches = (
            0
            if (
                self.fused_mlp_enabled_flag
                or self.fused_scaled_mlp_enabled_flag
            )
            else 1
        )
        per_layer = 1 + 1 + q_norm_present + k_norm_present + 1 + silu_mul_launches + 1
        if self.fused_residual_norm_enabled:
            per_layer -= 2
        return self.layers * per_layer + 1

    @property
    def attention_launches_per_token(self) -> int:
        return 0 if self.attention_mode == "current_only_smoke" else self.layers * 3

    @property
    def estimated_weight_read_bytes_per_token(self) -> int:
        # Account from the executable page contract instead of LayerPlan.
        # Recurrent Qwen3.5-family and Gemma-4 runtimes intentionally use
        # specialized graphs and do not create LayerPlan objects; returning
        # zero for them hid the dominant body traffic.
        specs = getattr(self.weights, "page_specs", {})
        total = 0
        for page in specs.values():
            if page.get("layer") is None:
                continue
            # Physical fusion containers and their logical children describe
            # the same bytes. Count logical execution pages only.
            if page.get("kind") == "fused_physical":
                continue
            total += int(page.get("size") or 0)

        head_spec = specs.get("lm_head.weight")
        if head_spec is not None:
            total += int(head_spec.get("size") or 0)
            scale_id = head_spec.get("scale_page")
            if scale_id in specs:
                total += int(specs[scale_id].get("size") or 0)
        else:
            # Tied embedding heads scan the full embedding matrix for logits.
            embed_spec = specs.get("model.embed_tokens.weight")
            if embed_spec is not None:
                total += int(embed_spec.get("size") or 0)
                scale_id = embed_spec.get("scale_page")
                if scale_id in specs:
                    total += int(specs[scale_id].get("size") or 0)

        final_norm = specs.get("model.norm.weight")
        if final_norm is not None:
            total += int(final_norm.get("size") or 0)
        if self._lm_head_scale is not None:
            # A runtime-created head scale is absent from page_specs. Archive
            # and PagePool scale pages were already counted above.
            head_scale_id = (
                head_spec.get("scale_page") if head_spec is not None else None
            )
            if head_scale_id not in specs:
                total += _tensor_nbytes(self._lm_head_scale)
            if self.lm_head_topk_guard > 0:
                exact_head = self._exact_lm_head()
                total += (
                    min(
                        self.lm_head_topk_guard,
                        int(exact_head.shape[0]),
                    )
                    * int(exact_head.shape[1])
                    * exact_head.element_size()
                )
        total += self.fp8_sparse_residual_bytes
        total += self.mxfp4_binary_residual_bytes
        total += self.mxfp4_int8_row_override_bytes
        total += self.mxfp4_row_postscale_bytes
        return total

    @property
    def resident_weight_bytes(self) -> int:
        lm_head_external = self.lm_head_external_resident_bytes
        if isinstance(self.weights, ThinGpuPagePool) and getattr(
            self.weights, "fp8_head_external_bytes_registered", False
        ):
            # register_external_resident_bytes() already made this allocation
            # part of page-pool resident_weight_bytes.
            lm_head_external = 0
        return (
            self.weights.resident_weight_bytes
            + lm_head_external
            + self.runtime_fusion_extra_bytes
            + self.fused_mlp_extra_bytes
            + self.fp8_sparse_residual_bytes
            + self.mxfp4_binary_residual_bytes
            + self.mxfp4_int8_row_override_bytes
            + self.mxfp4_row_postscale_bytes
            + self.adaptive_exact_resident_bytes
        )

    @property
    def temp_buffer_bytes(self) -> int:
        """Bytes owned by persistent decode scratch/logit buffers.

        Tensor views are deduplicated by storage pointer so sliced buffers are
        not double-counted. Weight and KV tensors are intentionally excluded.
        """
        tensors: list[torch.Tensor] = []
        backend_buffers = getattr(self.kernel_backend, "buffers", None)
        if backend_buffers is not None:
            tensors.extend(
                value
                for value in vars(backend_buffers).values()
                if isinstance(value, torch.Tensor)
            )
        if self.kernel_backend is not None:
            tensors.extend(
                value
                for value in vars(self.kernel_backend).values()
                if isinstance(value, torch.Tensor)
            )
        for value in (
            self._logits_buffer,
            self._lm_head_shortlist_rows,
            self._lm_head_shortlist_logits,
            self._lm_head_shortlist_bias,
            self._lm_head_shortlist_tie_ids,
            self._lm_head_vocab_sentinel,
            self._moe_gate_up_buffer,
            self._moe_down_buffer,
        ):
            if isinstance(value, torch.Tensor):
                tensors.append(value)
        seen: set[tuple[int, int]] = set()
        total = 0
        for tensor in tensors:
            storage = tensor.untyped_storage()
            key = (storage.data_ptr(), storage.nbytes())
            if key in seen:
                continue
            seen.add(key)
            total += storage.nbytes()
        return total

    @torch.inference_mode()
    def profile_lm_head_components(
        self,
        hidden: torch.Tensor,
        repeats: int = 20,
    ) -> dict[str, float | str]:
        """Measure standalone head matvec and argmax after decode timing."""
        if self.device.type != "cuda":
            return {}
        repeats = max(1, repeats)
        if self._lm_head_scale is not None and self.lm_head_topk_guard > 0:
            _ = self.next_token_tensor(hidden)
            torch.cuda.synchronize(self.device)
            head_start = torch.cuda.Event(enable_timing=True)
            head_end = torch.cuda.Event(enable_timing=True)
            head_start.record()
            for _ in range(repeats):
                _ = self.next_token_tensor(hidden)
            head_end.record()
            head_end.synchronize()
            return {
                "lm_head_time_ms": (
                    head_start.elapsed_time(head_end) / repeats
                ),
                "argmax_time_ms": 0.0,
                "lm_head_component_profile": (
                    "standalone_fp8_shortlist_bf16_verify"
                ),
            }
        logits = self.logits(hidden)
        _ = torch.argmax(logits)
        torch.cuda.synchronize(self.device)

        head_start = torch.cuda.Event(enable_timing=True)
        head_end = torch.cuda.Event(enable_timing=True)
        head_start.record()
        for _ in range(repeats):
            logits = self.logits(hidden)
        head_end.record()

        argmax_start = torch.cuda.Event(enable_timing=True)
        argmax_end = torch.cuda.Event(enable_timing=True)
        argmax_start.record()
        for _ in range(repeats):
            _ = torch.argmax(logits)
        argmax_end.record()
        argmax_end.synchronize()
        return {
            "lm_head_time_ms": head_start.elapsed_time(head_end) / repeats,
            "argmax_time_ms": argmax_start.elapsed_time(argmax_end) / repeats,
            "lm_head_component_profile": "standalone_post_decode",
        }

    @torch.inference_mode()
    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        head = self._lm_head()
        if self._logits_buffer is None or self._logits_buffer.numel() != head.shape[0]:
            self._logits_buffer = torch.empty(
                int(head.shape[0]),
                device=self.device,
                dtype=hidden.dtype,
            )
        choice = self._matvec_choice_by_tensor_id.get(id(head), "triton")
        if self.kernel_backend is not None and self.use_triton_matvec:
            if self._lm_head_scale is not None:
                # A guarded execution head is only an acceleration structure
                # for greedy selection. Public full logits stay BF16 so
                # validation and sampling never observe a mixed FP8/BF16
                # vector.
                if self.lm_head_topk_guard > 0:
                    exact_head = self._exact_lm_head()
                    result = self.kernel_backend.matvec(
                        exact_head,
                        hidden,
                        self._logits_buffer,
                    )
                    bias = _optional_tensor(self.weights, "lm_head.bias")
                    if bias is not None:
                        result.add_(bias)
                    return self._apply_final_logit_softcap(result)
                if self.lm_head_int4_enabled:
                    rows, cols, group_size = self._int4_metadata[id(head)]
                    result = self.kernel_backend.int4_scaled_matvec(
                        head,
                        self._lm_head_scale,
                        hidden,
                        self._logits_buffer,
                        rows=rows,
                        cols=cols,
                        group_size=group_size,
                    )
                else:
                    result = self.kernel_backend.scaled_matvec(
                        head,
                        self._lm_head_scale,
                        hidden,
                        self._logits_buffer,
                        config_name=(
                            choice if choice.startswith("triton_loop_") else None
                        ),
                    )
                bias = _optional_tensor(self.weights, "lm_head.bias")
                if bias is not None:
                    result.add_(bias)
                return self._apply_final_logit_softcap(result)
            if (
                choice in {"triton", "triton_basic"}
                or choice.startswith("triton_loop_")
                or choice.startswith("row_block_m")
            ):
                tuned_choice, block_m, num_warps = (
                    self._large_matvec_config(head, choice)
                )
                result = self.kernel_backend.matvec(
                    head,
                    hidden,
                    self._logits_buffer,
                    config_name=(
                        tuned_choice
                        if tuned_choice.startswith(
                            ("triton_loop_", "row_block_m")
                        )
                        else None
                    ),
                    block_m=block_m,
                    num_warps=num_warps,
                )
                bias = _optional_tensor(self.weights, "lm_head.bias")
                if bias is not None:
                    result.add_(bias)
                return self._apply_final_logit_softcap(result)
        result = torch.mv(head, hidden, out=self._logits_buffer)
        bias = _optional_tensor(self.weights, "lm_head.bias")
        if bias is not None:
            result.add_(bias)
        return self._apply_final_logit_softcap(result)

    def _apply_final_logit_softcap(
        self,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        if self.final_logit_softcap is None:
            return logits
        logits.div_(self.final_logit_softcap)
        logits.tanh_()
        logits.mul_(self.final_logit_softcap)
        return logits

    @torch.inference_mode()
    def topk(
        self,
        hidden: torch.Tensor,
        k: int = 5,
        exact: bool = False,
    ) -> list[dict[str, float | int]]:
        if exact:
            if self._bf16_lm_head_tensor is None:
                raise RuntimeError(
                    "exact top-k requested without a resident BF16 lm_head; "
                    "enable keep_bf16_lm_head"
                )
            logits = torch.mv(self._bf16_lm_head_tensor, hidden)
        else:
            logits = self.logits(hidden)
        values, indices = torch.topk(logits.float(), k=max(1, k))
        return [
            {"token_id": int(index), "logit": float(value)}
            for value, index in zip(values.detach().cpu(), indices.detach().cpu())
        ]

    @torch.inference_mode()
    def next_token(self, hidden: torch.Tensor) -> int:
        raise RuntimeError(
            "next_token() would synchronize CUDA. "
            "Use next_token_tensor() inside decode loops. "
            "Convert its GPU scalar result only after timing."
        )

    def next_token_tensor(self, hidden: torch.Tensor) -> torch.Tensor:
        if getattr(self, "_profiler_enabled", False) and getattr(self, "_current_step_profile", None) is not None:
            self._current_step_profile["lm_head_start"] = torch.cuda.Event(enable_timing=True)
            self._current_step_profile["lm_head_end"] = torch.cuda.Event(enable_timing=True)
            self._current_step_profile["lm_head_start"].record()

        head = self._lm_head()
        if self.kernel_backend is not None and self.use_triton_matvec:
            if (
                self._lm_head_scale is not None
                and self.lm_head_topk_guard > 0
            ):
                res = self._shortlist_next_token_tensor(hidden)
            elif self._lm_head_scale is not None:
                approximate_logits = self.logits(hidden)
                res = torch.argmax(approximate_logits)
            elif self.lm_head_argmax_mode == "triton_two_stage":
                res = self.kernel_backend.matvec_argmax_tensor(head, hidden)
            elif self.lm_head_argmax_mode == "triton_persistent":
                res = self.kernel_backend.persistent_vocab_block_matvec_argmax_tensor(head, hidden)
            elif self._matvec_choice_by_tensor_id.get(id(head)) in {
                "torch",
                "torch_mv",
                "torch_matmul",
            }:
                res = torch.argmax(self.logits(hidden))
            elif self._matvec_choice_by_tensor_id.get(id(head), "").startswith(
                "triton_loop_"
            ):
                choice = self._matvec_choice_by_tensor_id[id(head)]
                logits = self.kernel_backend.logits_buffer(
                    int(head.shape[0]), hidden.dtype
                )
                self.kernel_backend.matvec(
                    head,
                    hidden,
                    logits,
                    config_name=choice,
                )
                res = torch.argmax(logits)
            elif self._matvec_choice_by_tensor_id.get(id(head), "").startswith(
                "row_block_m"
            ):
                choice = self._matvec_choice_by_tensor_id[id(head)]
                logits = self.kernel_backend.logits_buffer(
                    int(head.shape[0]), hidden.dtype
                )
                self.kernel_backend.matvec(
                    head,
                    hidden,
                    logits,
                    config_name=choice,
                )
                res = torch.argmax(logits)
            else:
                choice = self._matvec_choice_by_tensor_id.get(
                    id(head),
                    "triton",
                )
                tuned_choice, block_m, num_warps = (
                    self._large_matvec_config(head, choice)
                )
                res = self.kernel_backend.matvec_argmax_tensor(
                    head,
                    hidden,
                    block_m=block_m,
                    num_warps=num_warps,
                    config_name=(
                        tuned_choice
                        if tuned_choice.startswith("triton_loop_")
                        else None
                    ),
                )
        else:
            res = torch.argmax(self.logits(hidden).float())

        if getattr(self, "_profiler_enabled", False) and getattr(self, "_current_step_profile", None) is not None:
            self._current_step_profile["lm_head_end"].record()

        return res

    def _exact_lm_head(self) -> torch.Tensor:
        exact_head = self._bf16_lm_head_tensor
        if exact_head is None and self.lm_head_tied_to_embeddings:
            exact_head = self._embed_weight
        if exact_head is None:
            raise RuntimeError(
                "FP8 shortlist verification requires a resident BF16 "
                "lm_head; enable --keep-bf16-lm-head"
            )
        return exact_head

    def _shortlist_next_token_tensor(
        self,
        hidden: torch.Tensor,
    ) -> torch.Tensor:
        assert self.kernel_backend is not None
        assert self._lm_head_scale is not None
        head = self._lm_head()
        shortlist_size = min(
            self.lm_head_topk_guard,
            int(head.shape[0]),
        )
        approximate_logits = self.kernel_backend.logits_buffer(
            int(head.shape[0]),
            hidden.dtype,
        )
        if self.lm_head_int4_enabled:
            rows, cols, group_size = self._int4_metadata[id(head)]
            self.kernel_backend.int4_scaled_matvec(
                head,
                self._lm_head_scale,
                hidden,
                approximate_logits,
                rows=rows,
                cols=cols,
                group_size=group_size,
            )
        else:
            self.kernel_backend.scaled_matvec(
                head,
                self._lm_head_scale,
                hidden,
                approximate_logits,
            )
        bias = _optional_tensor(self.weights, "lm_head.bias")
        if bias is not None:
            approximate_logits.add_(bias)
        shortlist = torch.topk(
            approximate_logits,
            shortlist_size,
            sorted=False,
        ).indices

        exact_head = self._exact_lm_head()
        if bias is None:
            return self.kernel_backend.indexed_matvec_argmax_tensor(
                exact_head,
                hidden,
                shortlist,
            )
        expected_rows = (shortlist_size, int(exact_head.shape[1]))
        if (
            self._lm_head_shortlist_rows is None
            or tuple(self._lm_head_shortlist_rows.shape) != expected_rows
        ):
            self._lm_head_shortlist_rows = torch.empty(
                expected_rows,
                device=self.device,
                dtype=exact_head.dtype,
            )
            self._lm_head_shortlist_logits = torch.empty(
                shortlist_size,
                device=self.device,
                dtype=hidden.dtype,
            )
            self._lm_head_shortlist_tie_ids = torch.empty(
                shortlist_size,
                device=self.device,
                dtype=shortlist.dtype,
            )
            self._lm_head_vocab_sentinel = torch.full(
                (),
                int(head.shape[0]),
                device=self.device,
                dtype=shortlist.dtype,
            )
            if bias is not None:
                self._lm_head_shortlist_bias = torch.empty(
                    shortlist_size,
                    device=self.device,
                    dtype=bias.dtype,
                )
        assert self._lm_head_shortlist_rows is not None
        assert self._lm_head_shortlist_logits is not None
        assert self._lm_head_shortlist_tie_ids is not None
        assert self._lm_head_vocab_sentinel is not None
        torch.index_select(
            exact_head,
            0,
            shortlist,
            out=self._lm_head_shortlist_rows,
        )
        torch.mv(
            self._lm_head_shortlist_rows,
            hidden,
            out=self._lm_head_shortlist_logits,
        )
        if bias is not None:
            assert self._lm_head_shortlist_bias is not None
            torch.index_select(
                bias,
                0,
                shortlist,
                out=self._lm_head_shortlist_bias,
            )
            self._lm_head_shortlist_logits.add_(
                self._lm_head_shortlist_bias
            )
        # torch.argmax returns the first (lowest vocab index) exact maximum.
        # torch.topk(sorted=False) does not preserve vocab order, so a local
        # argmax would choose a different token when BF16 logits tie.
        exact_max = torch.max(self._lm_head_shortlist_logits)
        torch.where(
            self._lm_head_shortlist_logits == exact_max,
            shortlist,
            self._lm_head_vocab_sentinel,
            out=self._lm_head_shortlist_tie_ids,
        )
        return torch.min(self._lm_head_shortlist_tie_ids)

    def _fused_qkv(
        self,
        layer: int,
        hidden: torch.Tensor,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        fused = self._fused_matrix(
            self.weights,
            f"layer_{layer}_attn_qkv_fused",
            _layer_tensor(layer, "self_attn.q_proj.weight"),
            [
                _layer_tensor(layer, "self_attn.q_proj.weight"),
                _layer_tensor(layer, "self_attn.k_proj.weight"),
                _layer_tensor(layer, "self_attn.v_proj.weight"),
            ],
        )
        if fused is None:
            return None
        projected = torch.mv(fused, hidden)
        q_end = self.q_dim
        k_end = q_end + self.kv_dim
        return projected[:q_end], projected[q_end:k_end], projected[k_end:]

    def _fused_gate_up(
        self,
        layer: int,
        hidden: torch.Tensor,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        gate_id = _layer_tensor(layer, "mlp.gate_proj.weight")
        up_id = _layer_tensor(layer, "mlp.up_proj.weight")
        fused = self._fused_matrix(
            self.weights,
            f"layer_{layer}_mlp_gate_up_fused",
            gate_id,
            [gate_id, up_id],
        )
        if fused is None:
            return None
        projected = torch.mv(fused, hidden)
        gate_rows = int(self.weights.tensor(gate_id).shape[0])
        return projected[:gate_rows], projected[gate_rows:]

    def _fused_matrix(
        self,
        weights: ThinGpuWeights | ThinGpuPagePool,
        fused_id: str,
        dtype_source_id: str,
        logical_ids: list[str],
    ) -> Optional[torch.Tensor]:
        if not self.evict_completed_layers and fused_id in self._fused_matrix_cache:
            return self._fused_matrix_cache[fused_id]
        matrix = _fused_matrix_for_layer(weights, fused_id, dtype_source_id, logical_ids)
        if matrix is not None and not self.evict_completed_layers:
            self._fused_matrix_cache[fused_id] = matrix
        return matrix

    def _copy_embed_token(
        self,
        embed: torch.Tensor,
        token_id: int | torch.Tensor,
        out: torch.Tensor,
    ) -> torch.Tensor:
        """
        Copy embedding row without CPU sync.

        If token_id is a GPU scalar returned by argmax, never call .item().
        """
        if isinstance(self.weights, ThinGpuPagePool) and self.weights.cpu_embed:
            return self.weights.copy_cpu_embedding_row(token_id, out)
        if isinstance(token_id, torch.Tensor):
            token = token_id.reshape(1).to(device=embed.device, dtype=torch.long)
            src = embed.index_select(0, token).reshape(-1)
            scale = (
                self._embed_scale.index_select(0, token)
                if self._embed_scale is not None
                else None
            )
        else:
            src = embed[token_id]
            scale = (
                self._embed_scale[token_id]
                if self._embed_scale is not None
                else None
            )
        out.copy_(src, non_blocking=True)
        if scale is not None:
            out.mul_(scale)
        return out

    def _forward_token_triton(
        self,
        embed: torch.Tensor,
        token_id: int | torch.Tensor,
        layer_count: int,
        token_index: int,
    ) -> torch.Tensor:
        backend = self.kernel_backend
        assert backend is not None
        self._active_token_index = token_index
        buffers = backend.buffers
        hidden = self._copy_embed_token(embed, token_id, buffers.hidden_a)
        hidden_slot_a = True
        precomputed_input_norm: Optional[torch.Tensor] = None
        final_norm_ready = False

        for layer in range(layer_count):
            self._capture_debug(layer, "layer_input", hidden)
            if getattr(self, "_profiler_enabled", False):
                lev = self._current_step_profile["layers"][layer] if layer < len(self._current_step_profile["layers"]) else None
                if lev is None:
                    lev = LayerProfileEvents()
                    self._current_step_profile["layers"].append(lev)
                lev.layer_start.record()
                lev.attn_start.record()
                lev.qkv_start.record()

            plan = (
                self._layer_plan[layer]
                if self._layer_plan is not None
                else self._make_layer_plan(layer)
            )
            for distance in range(1, self.prefetch_distance + 1):
                if hasattr(self.weights, "prefetch_layer"):
                    self.weights.prefetch_layer(layer + distance)

            if precomputed_input_norm is not None:
                normed = precomputed_input_norm
                precomputed_input_norm = None
            else:
                normed = self._runtime_norm(
                    hidden,
                    plan.input_layernorm_weight,
                    _layer_tensor(layer, "input_layernorm.weight"),
                    buffers.normed,
                )
            self._capture_debug(layer, "post_input_rmsnorm", normed)
            q, k, v = self._triton_qkv(plan, normed, lev if getattr(self, "_profiler_enabled", False) else None)
            for projected, suffix in (
                (q, "self_attn.q_proj.bias"),
                (k, "self_attn.k_proj.bias"),
                (v, "self_attn.v_proj.bias"),
            ):
                bias = _optional_tensor(
                    self.weights, _layer_tensor(layer, suffix)
                )
                if bias is not None:
                    projected.add_(bias)
            self._capture_debug(layer, "q_projection", q)
            self._capture_debug(layer, "k_projection", k)
            self._capture_debug(layer, "v_projection", v)
            if getattr(self, "_profiler_enabled", False):
                lev.qkv_end.record()
            if plan.q_norm is not None:
                q = backend.head_rms_norm(
                    q,
                    plan.q_norm,
                    buffers.q_norm,
                    self.heads,
                    self.head_dim,
                    self.rms_norm_eps,
                )
            if plan.k_norm is not None:
                k = backend.head_rms_norm(
                    k,
                    plan.k_norm,
                    buffers.k_norm,
                    self.kv_heads,
                    self.head_dim,
                    self.rms_norm_eps,
                )
            self._capture_debug(layer, "q_after_q_norm", q)
            self._capture_debug(layer, "k_after_k_norm", k)
            if self.attention_mode == "causal_kv":
                q, k = self._apply_rope(q, k, layer, token_index)
            self._capture_debug(layer, "q_after_rope", q)
            self._capture_debug(layer, "k_after_rope", k)
            if self.kv_cache is not None and not getattr(
                self, "_autotuning", False
            ):
                self.kv_cache.append(layer, k, v, token_index)

            if (
                self.attention_mode == "current_only_smoke"
                and
                self.use_triton_matvec
                and self._weight_scale(plan.o_proj) is None
                and self._matvec_choice_by_tensor_id.get(id(plan.o_proj), "triton")
                in {"triton", "triton_basic"}
            ):
                if getattr(self, "_profiler_enabled", False):
                    lev.o_proj_start.record()
                attn_out = backend.repeat_kv_matvec(
                    plan.o_proj,
                    v,
                    buffers.attn_out,
                    self.heads,
                    self.kv_heads,
                    self.head_dim,
                )
            else:
                attn = self._attention(
                    q,
                    k,
                    v,
                    layer,
                    token_index,
                    lev if getattr(self, "_profiler_enabled", False) else None,
                )
                if getattr(self, "_profiler_enabled", False):
                    lev.o_proj_start.record()
                attn_out = self._runtime_matvec(
                    plan.o_proj,
                    attn,
                    buffers.attn_out,
                )
            o_bias = _optional_tensor(
                self.weights,
                _layer_tensor(layer, "self_attn.o_proj.bias"),
            )
            if o_bias is not None:
                attn_out.add_(o_bias)
            self._capture_debug(layer, "o_proj_output", attn_out)
            if getattr(self, "_profiler_enabled", False):
                lev.o_proj_end.record()
            next_hidden = buffers.hidden_b if hidden_slot_a else buffers.hidden_a
            if self.fused_residual_norm_enabled:
                hidden, normed = backend.add_rms_norm(
                    hidden,
                    attn_out,
                    next_hidden,
                    plan.post_attention_layernorm_weight,
                    buffers.normed,
                    self.rms_norm_eps,
                )
            else:
                hidden = self._runtime_add(hidden, attn_out, next_hidden)
            self._capture_debug(layer, "post_attention_residual", hidden)
            hidden_slot_a = not hidden_slot_a
            if getattr(self, "_profiler_enabled", False):
                lev.attn_end.record()
                lev.mlp_start.record()

            if not self.fused_residual_norm_enabled:
                normed = self._runtime_norm(
                    hidden,
                    plan.post_attention_layernorm_weight,
                    _layer_tensor(
                        layer,
                        "post_attention_layernorm.weight",
                    ),
                    buffers.normed,
                )
            self._capture_debug(layer, "post_attention_rmsnorm", normed)

            # Determine whether to use a fused MLP projection/activation path.
            gate_id = _layer_tensor(layer, "mlp.gate_proj.weight")
            up_id = _layer_tensor(layer, "mlp.up_proj.weight")
            gate_bias = _optional_tensor(
                self.weights,
                _layer_tensor(layer, "mlp.gate_proj.bias"),
            )
            up_bias = _optional_tensor(
                self.weights,
                _layer_tensor(layer, "mlp.up_proj.bias"),
            )
            gate_scale = self._weight_scale(plan.gate_proj)
            up_scale = self._weight_scale(plan.up_proj)
            if gate_scale is not None and gate_scale.numel() == 1:
                gate_scale = gate_scale.expand(int(plan.gate_proj.shape[0]))
            if up_scale is not None and up_scale.numel() == 1:
                up_scale = up_scale.expand(int(plan.up_proj.shape[0]))
            if isinstance(self.weights, ThinGpuWeights):
                gate_up_already_loaded = True
            else:
                gate_up_already_loaded = gate_id in self.weights.tensors and up_id in self.weights.tensors

            use_fused_scaled = (
                self.fused_scaled_mlp_enabled_flag
                and self.kernel_backend is not None
                and gate_up_already_loaded
                and gate_scale is not None
                and up_scale is not None
                and gate_scale.ndim == 1
                and up_scale.ndim == 1
                and gate_bias is None
                and up_bias is None
            )
            if self.fused_mlp_enabled_flag:
                if self.kernel_backend is None:
                    self._fused_mlp_fallbacks += 1
                    use_fused = False
                elif not gate_up_already_loaded:
                    self._fused_mlp_fallbacks += 1
                    use_fused = False
                else:
                    use_fused = True
            else:
                use_fused = False

            if use_fused_scaled:
                if getattr(self, "_profiler_enabled", False):
                    lev.fused_mlp_start.record()
                activation = backend.fused_scaled_gate_up_silu(
                    plan.gate_proj,
                    gate_scale,
                    plan.up_proj,
                    up_scale,
                    normed,
                    buffers.mlp_act,
                )
                if getattr(self, "_profiler_enabled", False):
                    lev.fused_mlp_end.record()
            elif use_fused:
                if getattr(self, "_profiler_enabled", False):
                    lev.fused_mlp_start.record()
                activation = backend.fused_gate_up_silu(plan.gate_proj, plan.up_proj, normed, buffers.mlp_act)
                if getattr(self, "_profiler_enabled", False):
                    lev.fused_mlp_end.record()
            else:
                gate, up = self._triton_gate_up(plan, normed, lev if getattr(self, "_profiler_enabled", False) else None)
                if gate_bias is not None:
                    gate.add_(gate_bias)
                if up_bias is not None:
                    up.add_(up_bias)
                self._capture_debug(layer, "gate_projection", gate)
                self._capture_debug(layer, "up_projection", up)
                if getattr(self, "_profiler_enabled", False):
                    lev.silu_mul_start.record()
                activation = self._runtime_silu_mul(gate, up, buffers.mlp_act)
                self._capture_debug(layer, "gated_activation", activation)
                if getattr(self, "_profiler_enabled", False):
                    lev.silu_mul_end.record()

            if getattr(self, "_profiler_enabled", False):
                lev.down_proj_start.record()
            mlp = self._runtime_matvec(
                plan.down_proj,
                activation,
                buffers.mlp,
            )
            down_bias = _optional_tensor(
                self.weights,
                _layer_tensor(layer, "mlp.down_proj.bias"),
            )
            if down_bias is not None:
                mlp.add_(down_bias)
            self._capture_debug(layer, "down_projection", mlp)
            if getattr(self, "_profiler_enabled", False):
                lev.down_proj_end.record()
            next_hidden = buffers.hidden_b if hidden_slot_a else buffers.hidden_a
            if self.fused_residual_norm_enabled:
                if layer + 1 < layer_count:
                    next_plan = (
                        self._layer_plan[layer + 1]
                        if self._layer_plan is not None
                        else self._make_layer_plan(layer + 1)
                    )
                    hidden, precomputed_input_norm = backend.add_rms_norm(
                        hidden,
                        mlp,
                        next_hidden,
                        next_plan.input_layernorm_weight,
                        buffers.normed,
                        self.rms_norm_eps,
                    )
                else:
                    hidden, _ = backend.add_rms_norm(
                        hidden,
                        mlp,
                        next_hidden,
                        self._final_norm_weight,
                        buffers.final,
                        self.rms_norm_eps,
                    )
                    final_norm_ready = True
            else:
                hidden = self._runtime_add(hidden, mlp, next_hidden)
            self._capture_debug(layer, "final_residual_after_mlp", hidden)
            hidden_slot_a = not hidden_slot_a

            if self.evict_completed_layers and hasattr(self.weights, "evict_completed_layer"):
                self.weights.evict_completed_layer(layer)
            if getattr(self, "_profiler_enabled", False):
                lev.mlp_end.record()
                lev.layer_end.record()
            if self._layer_plan is None:
                del plan

        if final_norm_ready:
            return buffers.final
        return self._runtime_norm(
            hidden,
            self._final_norm_weight,
            "model.norm.weight",
            buffers.final,
        )

    def _can_runtime_fuse_weights(self) -> bool:
        # Runtime fusion duplicates weights in VRAM. It only gave a tiny speedup
        # and nearly doubled VRAM, so keep it opt-in.
        return (
            os.environ.get("THINTENSOR_RUNTIME_FUSE", "0") == "1"
            and isinstance(self.weights, ThinGpuWeights)
            and not self.evict_completed_layers
            and not self.qkv_fp8
            and not self.gate_up_fp8
        )

    def _runtime_fused_matrix(
        self,
        cache: dict[int, torch.Tensor],
        layer: int,
        ids: list[str],
    ) -> Optional[torch.Tensor]:
        if not self._can_runtime_fuse_weights():
            return None
        if layer in cache:
            return cache[layer]

        try:
            parts = [self.weights.tensor(page_id) for page_id in ids]
        except KeyError:
            return None

        if not parts:
            return None
        if any(part.dim() != 2 for part in parts):
            return None

        cols = int(parts[0].shape[1])
        dtype = parts[0].dtype
        device = parts[0].device

        if any(int(part.shape[1]) != cols for part in parts):
            return None
        if any(part.dtype != dtype or part.device != device for part in parts):
            return None

        # One-time GPU concat. This costs memory but removes 2 matvec launches for qkv
        # and 1 matvec launch for gate/up on every token.
        fused = torch.cat(parts, dim=0).contiguous()
        cache[layer] = fused
        return fused

    def _runtime_fused_qkv(self, layer: int) -> Optional[torch.Tensor]:
        return self._runtime_fused_matrix(
            self._runtime_fused_qkv_cache,
            layer,
            [
                _layer_tensor(layer, "self_attn.q_proj.weight"),
                _layer_tensor(layer, "self_attn.k_proj.weight"),
                _layer_tensor(layer, "self_attn.v_proj.weight"),
            ],
        )

    def _runtime_fused_gate_up(self, layer: int) -> Optional[torch.Tensor]:
        return self._runtime_fused_matrix(
            self._runtime_fused_gate_up_cache,
            layer,
            [
                _layer_tensor(layer, "mlp.gate_proj.weight"),
                _layer_tensor(layer, "mlp.up_proj.weight"),
            ],
        )

    def _triton_qkv(
        self,
        plan: LayerPlan,
        hidden: torch.Tensor,
        lev: Optional[LayerProfileEvents] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        backend = self.kernel_backend
        assert backend is not None
        buffers = backend.buffers
        fused = None if self.qkv_fp8 else plan.qkv_fused
        q = buffers.qkv[: self.q_dim]
        k = buffers.qkv[self.q_dim : self.q_dim + self.kv_dim]
        v = buffers.qkv[self.q_dim + self.kv_dim : self.q_dim + 2 * self.kv_dim]
        qkv_weights = (plan.q_proj, plan.k_proj, plan.v_proj)
        qkv_int4 = tuple(
            self._int4_metadata.get(id(weight))
            for weight in qkv_weights
        )
        adaptive_qkv = None
        if (
            self._adaptive_switch_token_index >= 0
            and self._active_token_index < self._adaptive_switch_token_index
        ):
            exact_weights = tuple(
                self._adaptive_exact_weights.get(id(weight))
                for weight in (plan.q_proj, plan.k_proj, plan.v_proj)
            )
            if all(weight is not None for weight in exact_weights):
                adaptive_qkv = exact_weights
        if all(
            meta is not None and meta[2] >= meta[1]
            for meta in qkv_int4
        ):
            assert all(meta is not None for meta in qkv_int4)
            scales = tuple(
                self._weight_scale(weight) for weight in qkv_weights
            )
            assert all(scale is not None for scale in scales)
            backend.multi_int4_scaled_matvec(
                qkv_weights,
                scales,
                tuple(meta[0] for meta in qkv_int4),
                qkv_int4[0][1],
                hidden,
                buffers.qkv,
            )
        elif (
            os.environ.get("THINTENSOR_TENSORCORE_MATVEC", "0") == "1"
            and os.environ.get("THINTENSOR_MULTI_TENSORCORE", "0") == "1"
            and all(
                self._weight_scale(weight) is not None
                and self._weight_scale(weight).ndim == 1
                for weight in qkv_weights
            )
            and len({weight.dtype for weight in qkv_weights}) == 1
            and qkv_weights[0].dtype
            in {torch.float8_e4m3fn, torch.int8}
        ):
            scales = tuple(
                self._weight_scale(weight) for weight in qkv_weights
            )
            assert all(scale is not None for scale in scales)
            backend.multi_scaled_tensorcore_matvec(
                qkv_weights,
                scales,
                tuple(int(weight.shape[0]) for weight in qkv_weights),
                int(qkv_weights[0].shape[1]),
                hidden,
                buffers.qkv,
            )
        elif adaptive_qkv is not None:
            backend.multi_matvec(
                adaptive_qkv,
                hidden,
                buffers.qkv,
            )
        elif fused is not None:
            self._runtime_matvec(fused, hidden, buffers.qkv)
        elif self.use_triton_matvec and not any(
            self._weight_scale(weight) is not None
            for weight in (plan.q_proj, plan.k_proj, plan.v_proj)
        ):
            backend.multi_matvec(
                (plan.q_proj, plan.k_proj, plan.v_proj),
                hidden,
                buffers.qkv,
            )
        else:
            self._runtime_matvec(plan.q_proj, hidden, q)
            self._runtime_matvec(plan.k_proj, hidden, k)
            self._runtime_matvec(plan.v_proj, hidden, v)
        return q, k, v

    def _triton_gate_up(
        self,
        plan: LayerPlan,
        hidden: torch.Tensor,
        lev: Optional[LayerProfileEvents] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        backend = self.kernel_backend
        assert backend is not None
        buffers = backend.buffers
        gate = buffers.gate_up[: self.intermediate_size]
        up = buffers.gate_up[self.intermediate_size : 2 * self.intermediate_size]
        gate_up_weights = (plan.gate_proj, plan.up_proj)
        gate_up_int4 = tuple(
            self._int4_metadata.get(id(weight))
            for weight in gate_up_weights
        )
        fused = None if self.gate_up_fp8 else plan.gate_up_fused
        if all(
            meta is not None and meta[2] >= meta[1]
            for meta in gate_up_int4
        ):
            assert all(meta is not None for meta in gate_up_int4)
            scales = tuple(
                self._weight_scale(weight)
                for weight in gate_up_weights
            )
            assert all(scale is not None for scale in scales)
            backend.multi_int4_scaled_matvec(
                gate_up_weights,
                scales,
                tuple(meta[0] for meta in gate_up_int4),
                gate_up_int4[0][1],
                hidden,
                buffers.gate_up,
            )
        elif (
            os.environ.get("THINTENSOR_TENSORCORE_MATVEC", "0") == "1"
            and os.environ.get("THINTENSOR_MULTI_TENSORCORE", "0") == "1"
            and all(
                self._weight_scale(weight) is not None
                and self._weight_scale(weight).ndim == 1
                for weight in gate_up_weights
            )
            and plan.gate_proj.dtype == plan.up_proj.dtype
            and plan.gate_proj.dtype
            in {torch.float8_e4m3fn, torch.int8}
        ):
            scales = tuple(
                self._weight_scale(weight) for weight in gate_up_weights
            )
            assert all(scale is not None for scale in scales)
            backend.multi_scaled_tensorcore_matvec(
                gate_up_weights,
                scales,
                tuple(
                    int(weight.shape[0]) for weight in gate_up_weights
                ),
                int(gate_up_weights[0].shape[1]),
                hidden,
                buffers.gate_up,
            )
        elif fused is not None:
            if lev is not None:
                lev.gate_proj_start.record()
            self._runtime_matvec(fused, hidden, buffers.gate_up)
            if lev is not None:
                lev.gate_proj_end.record()
        elif self.use_triton_matvec and not any(
            self._weight_scale(weight) is not None
            for weight in (plan.gate_proj, plan.up_proj)
        ):
            if lev is not None:
                lev.gate_proj_start.record()
            backend.multi_matvec(
                (plan.gate_proj, plan.up_proj),
                hidden,
                buffers.gate_up,
            )
            if lev is not None:
                lev.gate_proj_end.record()
        else:
            if lev is not None:
                lev.gate_proj_start.record()
            self._runtime_matvec(plan.gate_proj, hidden, gate)
            if lev is not None:
                lev.gate_proj_end.record()
                lev.up_proj_start.record()
            self._runtime_matvec(plan.up_proj, hidden, up)
            if lev is not None:
                lev.up_proj_end.record()
        return gate, up

    def _runtime_matvec(
        self,
        weight: torch.Tensor,
        x: torch.Tensor,
        out: torch.Tensor,
    ) -> torch.Tensor:
        rows = int(weight.shape[0])
        if weight.device.type == "cpu" and isinstance(
            self.weights, ThinGpuPagePool
        ):
            page_id = self.weights._tensor_id_to_page_id.get(id(weight))
            if page_id in self.weights._cpu_int4_pages:
                result = self.weights.cpu_int4_matvec(page_id, x, out)
                residual = self.weights._cpu_int4_residual_sidecars.get(page_id)
                if residual is not None:
                    values, indices = residual
                    self.kernel_backend.sparse_residual_matvec(
                        values, indices, x, result
                    )
                return result
        cols = int(weight.shape[1])
        if (rows < 128 or cols > 8192) and self._weight_scale(weight) is None:
            if out is None:
                out = torch.empty(rows, device=self.device, dtype=x.dtype)
            torch.mv(weight, x, out=out)
            self._record_streamed_tensor_use(weight)
            return out

        if isinstance(self.weights, ThinGpuPagePool):
            page_id = self.weights._tensor_id_to_page_id.get(id(weight))
            int4_cache = getattr(self.weights, "_int4_metadata_by_page", {})
            mxfp4_cache = getattr(self.weights, "_mxfp4_metadata_by_page", {})
            dense_lowbit_cache = getattr(
                self.weights, "_dense_lowbit_metadata_by_page", {}
            )
            if page_id in dense_lowbit_cache:
                self._dense_lowbit_metadata[id(weight)] = dense_lowbit_cache[page_id]
                self._int4_metadata.pop(id(weight), None)
                self._mxfp4_metadata.pop(id(weight), None)
            elif page_id in int4_cache:
                self._dense_lowbit_metadata.pop(id(weight), None)
                self._int4_metadata[id(weight)] = int4_cache[page_id]
                self._mxfp4_metadata.pop(id(weight), None)
            elif page_id in mxfp4_cache:
                self._dense_lowbit_metadata.pop(id(weight), None)
                metadata = mxfp4_cache[page_id]
                rows, cols = metadata[:2]
                hadamard_size = int(metadata[2]) if len(metadata) > 2 else 0
                self._mxfp4_metadata[id(weight)] = (rows, cols, hadamard_size)
                self._int4_metadata.pop(id(weight), None)
                signs_cache = getattr(self.weights, "_mxfp4_hadamard_signs_by_page", {})
                if page_id in signs_cache:
                    self._mxfp4_hadamard_signs[id(weight)] = signs_cache[page_id]
                else:
                    self._mxfp4_hadamard_signs.pop(id(weight), None)
                binary_cache = getattr(self.weights, "_mxfp4_binary_residuals_by_page", {})
                if page_id in binary_cache:
                    self._mxfp4_binary_residuals[id(weight)] = binary_cache[page_id]
                else:
                    self._mxfp4_binary_residuals.pop(id(weight), None)
            else:
                self._dense_lowbit_metadata.pop(id(weight), None)
                self._int4_metadata.pop(id(weight), None)
                self._mxfp4_metadata.pop(id(weight), None)
                self._mxfp4_hadamard_signs.pop(id(weight), None)
                self._mxfp4_binary_residuals.pop(id(weight), None)
        dense_lowbit_meta = self._dense_lowbit_metadata.get(id(weight))
        if dense_lowbit_meta is not None:
            assert self.kernel_backend is not None
            rows, original_cols, bits = dense_lowbit_meta
            scales = self._weight_scale(weight)
            if scales is None:
                raise RuntimeError("dense low-bit weight has no E8M0 scale page")
            bytes_per_group = 4 if bits == 1 else 8
            groups = original_cols // 32
            packed_view = weight.reshape(1, rows, groups, bytes_per_group)
            scale_view = scales.reshape(1, rows, groups)
            expert_index = getattr(self, "_dense_lowbit_expert_index", None)
            if expert_index is None or expert_index.device != weight.device:
                expert_index = torch.zeros(1, device=weight.device, dtype=torch.long)
                self._dense_lowbit_expert_index = expert_index
            output_view = out.reshape(1, rows)
            self.kernel_backend.mxfp_lowbit_selected_tensorcore_matvec(
                packed_view,
                scale_view,
                x,
                expert_index,
                output_view,
                bits=bits,
            )
            return out
        mxfp4_meta = self._mxfp4_metadata.get(id(weight))
        if mxfp4_meta is not None:
            assert self.kernel_backend is not None
            rows, cols, hadamard_size = mxfp4_meta
            scale = self._weight_scale(weight)
            assert scale is not None
            needs_sparse = (
                self.mxfp4_residual_terms > 0
                and id(weight) not in self._sparse_residual_sidecars
            )
            needs_postscale = (
                self.mxfp4_row_postscale
                and id(weight) not in self._mxfp4_row_postscales
            )
            if (
                (needs_sparse or needs_postscale)
                and isinstance(self.weights, ThinGpuPagePool)
            ):
                page_id = self.weights._tensor_id_to_page_id.get(id(weight), "")
                if page_id:
                    exact_cpu, _ = load_tensor_view(self.weights.archive, page_id)
                    sparse, postscale = self._build_mxfp4_sidecars_from_cpu(
                        exact_cpu,
                        weight,
                        scale,
                        hadamard_size=hadamard_size,
                        signs=self._mxfp4_hadamard_signs.get(id(weight)),
                        needs_sparse=needs_sparse,
                        needs_postscale=needs_postscale,
                    )
                    if sparse is not None:
                        self._sparse_residual_sidecars[id(weight)] = sparse
                    if postscale is not None:
                        self._mxfp4_row_postscales[id(weight)] = postscale
            original_x = x
            if hadamard_size:
                transformed_x = self._mxfp4_hadamard_buffers.get(cols)
                if transformed_x is None:
                    transformed_x = torch.empty(
                        cols,
                        device=x.device,
                        dtype=x.dtype,
                    )
                    self._mxfp4_hadamard_buffers[cols] = transformed_x
                self.kernel_backend.block_hadamard(
                    x,
                    transformed_x,
                    block_size=int(hadamard_size),
                    signs=self._mxfp4_hadamard_signs.get(id(weight)),
                )
                x = transformed_x
            binary_residual = self._mxfp4_binary_residuals.get(id(weight))
            result = self.kernel_backend.mxfp4_tensorcore_matvec(
                weight,
                scale,
                x,
                out,
                rows=rows,
                cols=cols,
                residual_bits=(
                    binary_residual[0]
                    if binary_residual is not None
                    else None
                ),
                residual_scales=(
                    binary_residual[1]
                    if binary_residual is not None
                    else None
                ),
                post_scales=self._mxfp4_row_postscales.get(id(weight)),
            )
            residual = self._sparse_residual_sidecars.get(id(weight))
            if residual is not None:
                values, indices = residual
                self.kernel_backend.sparse_residual_matvec(
                    values, indices, x, result
                )
            row_override = self._mxfp4_int8_row_overrides.get(id(weight))
            if row_override is not None:
                override_weight, override_scales, row_indices = row_override
                self.kernel_backend.selected_scaled_matvec(
                    override_weight,
                    override_scales,
                    row_indices,
                    original_x,
                    result,
                )
            return result
        int4_meta = self._int4_metadata.get(id(weight))
        if int4_meta is not None:
            assert self.kernel_backend is not None
            rows, cols, group_size = int4_meta
            scale = self._weight_scale(weight)
            assert scale is not None
            if (
                self._int4_sparse_residual_terms > 0
                and rows < 65536
                and id(weight) not in self._sparse_residual_sidecars
                and isinstance(self.weights, ThinGpuPagePool)
            ):
                page_id = self.weights._tensor_id_to_page_id.get(id(weight), "")
                if page_id:
                    residual_terms = (
                        self._int4_attention_residual_terms
                        if "self_attn." in page_id or "linear_attn." in page_id
                        else self._int4_sparse_residual_terms
                    )
                    exact_cpu, _ = load_tensor_view(self.weights.archive, page_id)
                    if residual_terms > 0:
                        self._sparse_residual_sidecars[id(weight)] = (
                            self._build_int4_residual_from_cpu(
                                exact_cpu,
                                weight,
                                scale,
                                terms=residual_terms,
                            )
                        )
            if (
                self._int4_lowrank_rank > 0
                and rows < 65536
                and id(weight) not in self._int4_lowrank_sidecars
                and isinstance(self.weights, ThinGpuPagePool)
            ):
                page_id = self.weights._tensor_id_to_page_id.get(id(weight), "")
                if page_id and "self_attn." not in page_id and "linear_attn." not in page_id:
                    exact_cpu, _ = load_tensor_view(self.weights.archive, page_id)
                    self._int4_lowrank_sidecars[id(weight)] = (
                        self._build_int4_lowrank_from_cpu(
                            exact_cpu,
                            weight,
                            scale,
                            rank=self._int4_lowrank_rank,
                        )
                    )
            result = self.kernel_backend.int4_scaled_matvec(
                weight,
                scale,
                x,
                out,
                rows=rows,
                cols=cols,
                group_size=group_size,
            )
            residual = self._sparse_residual_sidecars.get(id(weight))
            if residual is not None:
                values, indices = residual
                self.kernel_backend.sparse_residual_matvec(
                    values, indices, x, result
                )
            lowrank = self._int4_lowrank_sidecars.get(id(weight))
            if lowrank is not None:
                u, v = lowrank
                coefficients = torch.mv(v, x)
                result.addmv_(u, coefficients)
            return result

        adaptive_exact = None
        if (
            self._adaptive_switch_token_index >= 0
            and self._active_token_index < self._adaptive_switch_token_index
        ):
            adaptive_exact = self._adaptive_exact_weights.get(id(weight))

        # Page-pool tensors are destroyed and recreated as layers stream
        # through VRAM. Python object ids are therefore not stable dispatch
        # keys and may be reused by a different projection after eviction.
        # Shape/dtype/stride remains stable across reloads.
        if isinstance(self.weights, ThinGpuPagePool):
            # Page identity is stable even though the CUDA tensor object is
            # recreated. Apply explicit role overrides from that identity.
            page_id = self.weights._tensor_id_to_page_id.get(id(weight), "")
            choice = None
            if page_id:
                if page_id in {"lm_head.weight", "model.embed_tokens.weight"}:
                    choice = self.lm_head_backend_override
                elif any(
                    suffix in page_id
                    for suffix in ("mlp.gate_proj.weight", "mlp.up_proj.weight")
                ):
                    choice = self.gate_up_backend_override
                elif "mlp.down_proj.weight" in page_id:
                    choice = self.down_proj_backend_override
                elif any(
                    suffix in page_id
                    for suffix in (
                        "self_attn.q_proj.weight",
                        "self_attn.k_proj.weight",
                        "self_attn.v_proj.weight",
                        "self_attn.o_proj.weight",
                    )
                ):
                    choice = self.attn_proj_backend_override
            if choice is None:
                choice = self._matvec_backend_choices.get(
                    self._matvec_shape_key(weight)
                )
        else:
            # All-resident tensor ids are stable and carry explicit per-role
            # overrides; fall back to the generated per-shape winner table.
            choice = self._matvec_choice_by_tensor_id.get(id(weight))
            if choice is None:
                choice = self._matvec_backend_choices.get(
                    self._matvec_shape_key(weight)
                )
        if adaptive_exact is not None:
            choice = self._adaptive_exact_choices[id(weight)]
            weight = adaptive_exact
        scale = self._weight_scale(weight)
        tuned_choice, block_m, num_warps = self._large_matvec_config(
            weight,
            choice or "triton",
        )
        if (
            self.split_k_down_proj_enabled
            and self.kernel_backend is not None
            and int(weight.shape[0]) == self.hidden_size
            and int(weight.shape[1]) == self.intermediate_size
            and (scale is None or scale.ndim == 1)
        ):
            return self.kernel_backend.split_k_matvec(
                weight,
                x,
                out,
                scales=scale,
                split_k=2,
                block_m=32,
                block_n=256,
                num_warps=4,
            )

        if scale is not None and choice in {None, "torch_mv", "torch_matmul"}:
            # Materializing a full dequantized matrix is both slower and can
            # exceed VRAM for fused projection pages. Scaled weights always
            # stay on the streaming Triton path.
            choice = "triton"
        if scale is not None:
            if choice == "torch_mv" or choice is None:
                dequant = (
                    weight.to(dtype=x.dtype)
                    * scale.to(dtype=x.dtype)[:, None]
                )
                torch.mv(dequant, x, out=out)
                self._record_streamed_tensor_use(weight, scale)
                return out
            elif choice == "torch_matmul":
                dequant = (
                    weight.to(dtype=x.dtype)
                    * scale.to(dtype=x.dtype)[:, None]
                )
                torch.matmul(dequant, x, out=out)
                self._record_streamed_tensor_use(weight, scale)
                return out
            elif choice == "triton":
                assert self.kernel_backend is not None
                result = self.kernel_backend.scaled_matvec(
                    weight,
                    scale,
                    x,
                    out,
                    block_m=block_m,
                    num_warps=num_warps,
                )
                residual = self._sparse_residual_sidecars.get(id(weight))
                if residual is not None:
                    values, indices = residual
                    self.kernel_backend.sparse_residual_matvec(
                        values, indices, x, result
                    )
                self._record_streamed_tensor_use(weight, scale)
                return result
            elif choice.startswith("triton_loop_"):
                assert self.kernel_backend is not None
                result = self.kernel_backend.scaled_matvec(
                    weight,
                    scale,
                    x,
                    out,
                    config_name=tuned_choice,
                    block_m=block_m,
                    num_warps=num_warps,
                )
                residual = self._sparse_residual_sidecars.get(id(weight))
                if residual is not None:
                    values, indices = residual
                    self.kernel_backend.sparse_residual_matvec(
                        values, indices, x, result
                    )
                self._record_streamed_tensor_use(weight, scale)
                return result
            else:
                dequant = (
                    weight.to(dtype=x.dtype)
                    * scale.to(dtype=x.dtype)[:, None]
                )
                torch.mv(dequant, x, out=out)
                self._record_streamed_tensor_use(weight, scale)
                return out

        if choice is None:
            if self.kernel_backend is not None and self.use_triton_matvec:
                result = self.kernel_backend.matvec(
                    weight,
                    x,
                    out,
                    block_m=block_m,
                    num_warps=num_warps,
                )
                self._record_streamed_tensor_use(weight)
                return result
            torch.mv(weight, x, out=out)
            self._record_streamed_tensor_use(weight)
            return out

        if choice == "torch_mv":
            torch.mv(weight, x, out=out)
            self._record_streamed_tensor_use(weight)
            return out
        elif choice == "torch_matmul":
            torch.matmul(weight, x, out=out)
            self._record_streamed_tensor_use(weight)
            return out
        elif choice == "triton":
            assert self.kernel_backend is not None
            result = self.kernel_backend.matvec(
                weight,
                x,
                out,
                block_m=block_m,
                num_warps=num_warps,
            )
            self._record_streamed_tensor_use(weight)
            return result
        elif choice.startswith("triton_loop_"):
            assert self.kernel_backend is not None
            result = self.kernel_backend.matvec(
                weight,
                x,
                out,
                config_name=tuned_choice,
                block_m=block_m,
                num_warps=num_warps,
            )
            self._record_streamed_tensor_use(weight)
            return result
        else:
            torch.mv(weight, x, out=out)
            self._record_streamed_tensor_use(weight)
            return out

    def _record_streamed_tensor_use(
        self,
        *tensors: torch.Tensor | None,
    ) -> None:
        if not isinstance(self.weights, ThinGpuPagePool):
            return
        stream: torch.cuda.Stream | None = None
        event: torch.cuda.Event | None = None
        for tensor in tensors:
            if tensor is None or not isinstance(tensor, torch.Tensor) or not tensor.is_cuda:
                continue
            page_id = self.weights._tensor_id_to_page_id.get(id(tensor))
            if page_id is None:
                continue
            if stream is None:
                stream = torch.cuda.current_stream(tensor.device)
                event = torch.cuda.Event()
                event.record(stream)
            tensor.record_stream(stream)
            self.weights._page_use_events[page_id] = event

    def _large_matvec_config(
        self,
        weight: torch.Tensor,
        choice: str,
    ) -> tuple[str, int | None, int | None]:
        rows = int(weight.shape[0])
        cols = int(weight.shape[1])
        scaled = self._weight_scale(weight) is not None
        if not self.tuned_large_matvec_enabled:
            return choice, None, None
        if rows >= 65536 and cols >= 1024 and not scaled:
            return choice, 32, 4
        if rows >= cols * 4 and scaled:
            return choice, 8, 4
        if cols >= rows * 4:
            return choice, 32, 4 if scaled else 8
        if rows == cols and not scaled:
            return choice, 4, 4
        return choice, None, None

    def _weight_scale(self, weight: torch.Tensor) -> Optional[torch.Tensor]:
        if isinstance(self.weights, ThinGpuPagePool):
            page_id = self.weights._tensor_id_to_page_id.get(id(weight))
            if page_id is not None:
                scale_id = page_id + ".scale"
                if scale_id in self.weights.page_specs:
                    return self.weights.tensor(scale_id)
            return None
        return self._weight_scales.get(id(weight))

    def _runtime_add(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        out: torch.Tensor,
    ) -> torch.Tensor:
        backend = self.kernel_backend
        assert backend is not None
        if self.use_triton_elementwise:
            return backend.add(left, right, out)
        torch.add(left, right, out=out)
        return out

    def _normalization(
        self,
        x: torch.Tensor,
        weight_id: str,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.norm_kind == "rms_norm":
            weight = self.weights.tensor(weight_id)
            if self.kernel_backend is not None:
                if out is None:
                    out = torch.empty_like(x)
                return self.kernel_backend.rms_norm(
                    x,
                    weight,
                    out,
                    self.norm_eps,
                    weight_offset=self.norm_weight_offset,
                )
            if self.norm_weight_offset:
                cache_key = f"{weight_id}_offsetted"
                if getattr(self, "_use_tensor_cache", False):
                    try:
                        weight = self.weights._cached_tensors[cache_key]
                    except KeyError:
                        orig = self.weights.tensor(weight_id)
                        weight = (orig.float() + self.norm_weight_offset).to(dtype=x.dtype)
                        self.weights._cached_tensors[cache_key] = weight
                else:
                    orig = self.weights.tensor(weight_id)
                    weight = (orig.float() + self.norm_weight_offset).to(dtype=x.dtype)
                work = x.float()
                normed = work * torch.rsqrt(work.pow(2).mean() + self.norm_eps)
                return normed.to(dtype=x.dtype) * weight
            else:
                return _rms_norm(
                    x,
                    weight,
                    self.norm_eps,
                    weight_offset=0.0,
                )
        weight = self.weights.tensor(weight_id)
        bias = _optional_tensor(
            self.weights,
            _get_bias_id(weight_id),
        )
        return torch.nn.functional.layer_norm(
            x,
            (int(x.shape[-1]),),
            weight,
            bias,
            self.norm_eps,
        )

    def _runtime_norm(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        weight_id: str,
        out: torch.Tensor,
    ) -> torch.Tensor:
        backend = self.kernel_backend
        assert backend is not None
        if self.norm_kind == "rms_norm":
            return backend.rms_norm(
                x,
                weight,
                out,
                self.norm_eps,
                weight_offset=self.norm_weight_offset,
            )
        bias = _optional_tensor(
            self.weights,
            weight_id.removesuffix(".weight") + ".bias",
        )
        value = torch.nn.functional.layer_norm(
            x,
            (int(x.shape[-1]),),
            weight,
            bias,
            self.norm_eps,
        )
        out.copy_(value)
        return out

    def _runtime_silu_mul(
        self,
        gate: torch.Tensor,
        up: torch.Tensor,
        out: torch.Tensor,
    ) -> torch.Tensor:
        backend = self.kernel_backend
        assert backend is not None
        if self.use_triton_elementwise:
            return backend.silu_mul(gate, up, out)
        torch.mul(torch.nn.functional.silu(gate), up, out=out)
        return out

    def capture_cuda_graph(self, steps: int) -> None:
        if self.device.type != "cuda":
            return
        if self.attention_mode == "causal_kv":
            self._cuda_graph_error = (
                "causal_kv CUDA graph capture is disabled: the current graph "
                "hardcodes token_index=0 and would replay an invalid one-token "
                "attention history"
            )
            self._cuda_graphs_enabled = False
            return
        
        if self.kv_cache is not None:
            # Reconfigure block size to be large enough to hold all steps + warmup steps
            self.kv_cache.block_size = max(self.kv_cache.block_size, steps + 20)
            self.kv_cache.current.clear()
            self.kv_cache.blocks.clear()
            self.kv_cache.blocks_by_layer.clear()
            self.kv_cache.bytes = 0

        self._static_token_id = torch.zeros((), device=self.device, dtype=torch.long)
        self._static_hidden = None
        
        try:
            # Warm up
            hidden = self.forward_token(self._static_token_id, token_index=0)
            self._static_next_token = self.next_token_tensor(hidden).clone()
            torch.cuda.synchronize()
            
            # Start graph capture
            capture_start = time.perf_counter()
            self._cuda_graph = torch.cuda.CUDAGraph()
            
            with torch.cuda.graph(self._cuda_graph):
                hidden = self.forward_token(self._static_token_id, token_index=0)
                self._static_hidden = hidden
                next_token = self.next_token_tensor(hidden)
                self._static_next_token.copy_(next_token)
            
            torch.cuda.synchronize()
            self._cuda_graph_capture_s = time.perf_counter() - capture_start
            self._cuda_graphs_enabled = True
        except Exception as exc:
            self._cuda_graph_error = str(exc)
            self._cuda_graphs_enabled = False
            self._cuda_graph = None

    def replay_cuda_graph(self, token_id: torch.Tensor) -> torch.Tensor:
        self._static_token_id.copy_(token_id)
        self._cuda_graph.replay()
        return self._static_next_token



# Backward-compatible name retained for existing integrations.
ThinGpuQwenRuntime = ThinGpuCausalLMRuntime


class TempGuard:
    def __init__(self, device: str, max_temp: int = 87, poll_sec: float = 0.5) -> None:
        self.device = device
        self.max_temp = max_temp
        self.poll_sec = poll_sec
        self.peak_temp: int | None = None
        self.peak_power_w: float | None = None
        self.peak_memory_used_mib: int | None = None
        self.peak_utilization_pct: int | None = None
        self.samples = 0
        self.too_hot = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self.device != "cuda" or not _has_nvidia_smi():
            return
        self.sample()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def raise_if_hot(self, where: str) -> None:
        if self._thread is None:
            self.sample()
        if self.too_hot:
            raise RuntimeError(
                f"GPU temperature reached {self.peak_temp}C {where}; limit is {self.max_temp}C"
            )

    def _run(self) -> None:
        while not self._stop.wait(self.poll_sec):
            self.sample()

    def sample(self) -> None:
        sample = _read_gpu_sample()
        if sample is None:
            return
        self.samples += 1
        temp = sample["temperature_c"]
        self.peak_temp = temp if self.peak_temp is None else max(self.peak_temp, temp)
        power = sample["power_w"]
        if power is not None:
            self.peak_power_w = (
                power
                if self.peak_power_w is None
                else max(self.peak_power_w, power)
            )
        memory = sample["memory_used_mib"]
        if memory is not None:
            self.peak_memory_used_mib = (
                memory
                if self.peak_memory_used_mib is None
                else max(self.peak_memory_used_mib, memory)
            )
        utilization = sample["utilization_pct"]
        if utilization is not None:
            self.peak_utilization_pct = (
                utilization
                if self.peak_utilization_pct is None
                else max(self.peak_utilization_pct, utilization)
            )
        if temp >= self.max_temp:
            self.too_hot = True


def benchmark_gpu_runtime(
    archive_path: str | Path,
    token_id: int,
    steps: int,
    device: str = "cuda",
    dtype: Optional[torch.dtype] = torch.bfloat16,
    layers: Optional[int] = None,
    top_k: int = 5,
    max_gpu_temp: int = 87,
) -> dict[str, Any]:
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    rss0 = _rss_bytes()
    guard = TempGuard(device, max_gpu_temp)
    guard.start()
    try:
        load_start = time.perf_counter()
        weights = ThinGpuWeights(archive_path, device=device, dtype=dtype)
        model = weights.manifest["model"]
        model_heads = int(model["heads"])
        cache = PagedKVCache(
            layers=int(model["layers"]),
            kv_heads=int(model.get("kv_heads") or model_heads),
            head_dim=int(
                model.get("head_dim")
                or int(model["hidden_size"]) // model_heads
            ),
            device=weights.device,
            dtype=dtype or weights.tensor("model.norm.weight").dtype,
            old_codec="bf16",
        )
        runtime = ThinGpuCausalLMRuntime(weights, kv_cache=cache)
        if device == "cuda":
            torch.cuda.synchronize()
        load_s = time.perf_counter() - load_start

        guard.raise_if_hot("before first forward")
        first_start = time.perf_counter()
        hidden = runtime.forward_token(
            token_id,
            layers=layers,
            token_index=0,
        )
        if device == "cuda":
            torch.cuda.synchronize()
        first_s = time.perf_counter() - first_start

        guard.raise_if_hot("before timed loop")
        timed_start = time.perf_counter()
        for step in range(max(1, steps)):
            hidden = runtime.forward_token(
                token_id,
                layers=layers,
                token_index=step + 1,
            )
        if device == "cuda":
            torch.cuda.synchronize()
        forward_s = time.perf_counter() - timed_start
        top_logits = runtime.topk(hidden, top_k)
        rss1 = _rss_bytes()
        stats = weights.stats
        result = {
            "mode": "thin_gpu_native",
            "archive": str(archive_path),
            "device": device,
            "dtype": str(dtype).replace("torch.", "") if dtype is not None else "archive",
            "token_id": token_id,
            "layers": layers,
            "steps": max(1, steps),
            "load_s": load_s,
            "archive_open_s": stats.archive_open_s,
            "gpu_weight_load_s": stats.gpu_load_s,
            "first_forward_s": first_s,
            "forward_loop_s": forward_s,
            "forward_per_s": max(1, steps) / forward_s if forward_s > 0 else 0.0,
            "pages_loaded": stats.pages_loaded,
            "physical_pages_loaded": stats.physical_pages_loaded,
            "aliased_pages": stats.aliased_pages,
            "fused_logical_pages": stats.fused_logical_pages,
            "unique_gpu_weight_bytes": stats.unique_gpu_weight_bytes,
            "physical_weight_bytes": stats.physical_weight_bytes,
            "cpu_staging_bytes": stats.cpu_staging_bytes,
            "gpu_transfer_bytes": stats.gpu_transfer_bytes,
            "disk_read_s": stats.disk_read_s,
            "cpu_stage_s": stats.cpu_stage_s,
            "gpu_transfer_s": stats.gpu_transfer_s,
            "minor_page_faults": stats.minor_page_faults,
            "major_page_faults": stats.major_page_faults,
            "rss_delta_bytes": None if rss0 is None or rss1 is None else rss1 - rss0,
            "gpu_peak_allocated_bytes": _gpu_peak_allocated(device),
            "gpu_peak_reserved_bytes": _gpu_peak_reserved(device),
            "gpu_peak_temp_c": guard.peak_temp,
            "gpu_peak_power_w": guard.peak_power_w,
            "gpu_peak_memory_used_mib": guard.peak_memory_used_mib,
            "gpu_peak_utilization_pct": guard.peak_utilization_pct,
            "gpu_telemetry_samples": guard.samples,
            "thermal_stop": guard.too_hot,
            "attention_mode": runtime.attention_mode,
            "kv_cache": cache.telemetry(),
            "top_logits": top_logits,
        }
        weights.close()
        return result
    finally:
        guard.stop()


def _optional_tensor(
    weights: ThinGpuWeights | ThinGpuPagePool,
    page_id: str,
) -> Optional[torch.Tensor]:
    if getattr(weights, "_use_tensor_cache", False):
        try:
            return weights._cached_optional_tensors[page_id]
        except KeyError:
            pass
    if hasattr(weights, "has_page") and weights.has_page(page_id):
        val = weights.tensor(page_id)
    else:
        val = weights.tensors.get(page_id)
    if getattr(weights, "_use_tensor_cache", False):
        weights._cached_optional_tensors[page_id] = val
    return val


def _has_page(
    weights: ThinGpuWeights | ThinGpuPagePool,
    page_id: str,
) -> bool:
    if hasattr(weights, "has_page"):
        return bool(weights.has_page(page_id))
    return page_id in weights.tensors


def _first_optional_tensor(
    weights: ThinGpuWeights | ThinGpuPagePool,
    page_ids: tuple[str, ...],
) -> Optional[torch.Tensor]:
    for page_id in page_ids:
        tensor = _optional_tensor(weights, page_id)
        if tensor is not None:
            return tensor
    return None


def _dequantize_mxfp4(
    blocks: torch.Tensor,
    scales: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Dequantize selected MXFP4 expert blocks without expanding all experts."""
    if blocks.shape[:-1] != scales.shape:
        raise RuntimeError(
            f"MXFP4 blocks/scales mismatch: {blocks.shape} vs {scales.shape}"
        )
    fp4_values = torch.tensor(
        (
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ),
        dtype=dtype,
        device=blocks.device,
    )
    packed = blocks.to(torch.uint8)
    exponents = scales.to(torch.int32) - 127
    low = fp4_values[(packed & 0x0F).to(torch.long)]
    high = fp4_values[(packed >> 4).to(torch.long)]
    unpacked = torch.empty(
        (*packed.shape[:-1], packed.shape[-1] * 2),
        dtype=dtype,
        device=blocks.device,
    )
    unpacked[..., 0::2] = low
    unpacked[..., 1::2] = high
    unpacked = torch.ldexp(unpacked, exponents.unsqueeze(-1))
    flattened = unpacked.flatten(-2)
    if flattened.ndim < 3:
        raise RuntimeError(
            f"MXFP4 expert tensor must have at least 3 dimensions: {blocks.shape}"
        )
    return flattened.transpose(1, 2).contiguous()


def _fused_matrix_for_layer(
    weights: ThinGpuWeights | ThinGpuPagePool,
    fused_id: str,
    dtype_source_id: str,
    logical_ids: list[str],
) -> Optional[torch.Tensor]:
    page_specs = getattr(weights, "page_specs", {})
    if fused_id not in page_specs or any(page_id not in page_specs for page_id in logical_ids):
        return None
    specs = [page_specs[page_id] for page_id in logical_ids]
    if any(len(spec["shape"]) != 2 for spec in specs):
        return None
    cols = int(specs[0]["shape"][1])
    if any(int(spec["shape"][1]) != cols for spec in specs):
        return None
    source_dtype = map_dtype(page_specs[dtype_source_id]["dtype"])
    elem_size = torch.empty((), dtype=source_dtype).element_size()
    rows = sum(int(spec["shape"][0]) for spec in specs)
    byte_len = rows * cols * elem_size
    raw = weights.tensor(fused_id)
    if raw.dtype != torch.uint8 or raw.numel() < byte_len:
        return None
    matrix = raw.narrow(0, 0, byte_len).view(source_dtype).reshape(rows, cols)
    target_dtype = _target_dtype(source_dtype, getattr(weights, "dtype", None))
    if target_dtype != source_dtype:
        matrix = matrix.to(dtype=target_dtype)
    return matrix


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
            "linear_attn.in_proj_qkv.weight",
            "linear_attn.in_proj_z.weight",
            "linear_attn.out_proj.weight",
        )
    ) and ".experts." not in page_id


def _is_separate_expert_page(page_id: str) -> bool:
    def has_numbered_expert(marker: str) -> bool:
        if marker not in page_id:
            return False
        tail = page_id.split(marker, 1)[1]
        expert_id = tail.split(".", 1)[0]
        return expert_id.isdigit()

    return (
        has_numbered_expert(".mlp.experts.")
        or has_numbered_expert(".block_sparse_moe.experts.")
        or (
            page_id.startswith("__runtime__.layer.")
            and ".experts." in page_id
        )
    )


def _is_packed_expert_stream_page(page_id: str) -> bool:
    if ".mlp.experts." not in page_id and ".block_sparse_moe.experts." not in page_id:
        return False
    leaf = page_id.rsplit(".", 1)[-1]
    return leaf.endswith("_blocks") or leaf.endswith("_scales")


def _is_separate_expert_weight(page_id: str) -> bool:
    return _is_separate_expert_page(page_id) and page_id.endswith(".weight")


def _separate_expert_tensor_ids(
    weights: ThinGpuWeights | ThinGpuPagePool,
    layer: int,
    expert: int,
) -> tuple[str, str, str]:
    candidates = (
        (
            f"model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight",
            f"model.layers.{layer}.mlp.experts.{expert}.up_proj.weight",
            f"model.layers.{layer}.mlp.experts.{expert}.down_proj.weight",
        ),
        (
            f"model.layers.{layer}.block_sparse_moe.experts.{expert}.w1.weight",
            f"model.layers.{layer}.block_sparse_moe.experts.{expert}.w3.weight",
            f"model.layers.{layer}.block_sparse_moe.experts.{expert}.w2.weight",
        ),
    )
    for names in candidates:
        if all(
            (
                weights.has_page(name)
                if hasattr(weights, "has_page")
                else name in weights.tensors
            )
            for name in names
        ):
            return names
    raise RuntimeError(
        f"layer {layer} selected expert {expert}, but no supported separate "
        "gate/up/down tensor triplet exists"
    )


def _rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    weight_offset: float = 0.0,
) -> torch.Tensor:
    work = x.float()
    normed = work * torch.rsqrt(work.pow(2).mean() + eps)
    if weight_offset:
        return (normed * (weight.float() + weight_offset)).to(dtype=x.dtype)
    return normed.to(dtype=x.dtype) * weight


def _head_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    heads: int,
    head_dim: int,
    eps: float,
) -> torch.Tensor:
    shaped = x.reshape(heads, head_dim)
    work = shaped.float()
    normed = (work * torch.rsqrt(work.pow(2).mean(dim=-1, keepdim=True) + eps)).to(dtype=x.dtype)
    return (normed * weight.reshape(1, head_dim)).reshape(-1)


def _single_token_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    heads: int,
    kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    # For a no-cache one-token forward, attention softmax has one key, so the
    # output is exactly V repeated across grouped-query heads. Q/K are still
    # computed by the caller because real decode needs them for the KV cache.
    _ = q
    _ = k
    group = heads // kv_heads
    return v.reshape(kv_heads, head_dim).repeat_interleave(group, dim=0).reshape(heads * head_dim)


_LAYER_TENSOR_CACHE = {}

def _layer_tensor(layer: int, suffix: str) -> str:
    key = (layer, suffix)
    try:
        return _LAYER_TENSOR_CACHE[key]
    except KeyError:
        val = f"model.layers.{layer}.{suffix}"
        _LAYER_TENSOR_CACHE[key] = val
        return val


_BIAS_SUFFIX_CACHE = {}

def _get_bias_id(weight_id: str) -> str:
    try:
        return _BIAS_SUFFIX_CACHE[weight_id]
    except KeyError:
        val = weight_id.removesuffix(".weight") + ".bias"
        _BIAS_SUFFIX_CACHE[weight_id] = val
        return val


def _codec_bytes(codec: str) -> float:
    match codec.lower():
        case "q2":
            return 0.25
        case "q3":
            return 0.375
        case "q4" | "nvfp4":
            return 0.5
        case "q5":
            return 0.625
        case "q6":
            return 0.75
        case "q8" | "fp8":
            return 1.0
        case "fp16" | "bf16" | "high_precision":
            return 2.0
        case "fp32":
            return 4.0
        case _:
            return 2.0


def _parse_layer_selection(spec: Optional[str], layers: int) -> set[int]:
    if spec is None or not spec.strip() or spec.strip().lower() == "all":
        return set(range(layers))
    selected: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            start_text, end_text = part.split(":", 1)
            start = int(start_text) if start_text else 0
            end = int(end_text) if end_text else layers
            selected.update(range(start, end))
        else:
            selected.add(int(part))
    invalid = sorted(layer for layer in selected if layer < 0 or layer >= layers)
    if invalid:
        raise ValueError(
            f"FP8 layer selection contains out-of-range layers {invalid}; "
            f"model has {layers} layers"
        )
    return selected


def _rss_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:
        try:
            with open("/proc/self/status", "r", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) * 1024
        except OSError:
            return None
    return None


def _gpu_peak_allocated(device: str) -> int | None:
    if device != "cuda":
        return None
    return int(torch.cuda.max_memory_allocated())


def _gpu_peak_reserved(device: str) -> int | None:
    if device != "cuda":
        return None
    return int(torch.cuda.max_memory_reserved())


def _has_nvidia_smi() -> bool:
    return subprocess.run(
        ["bash", "-lc", "command -v nvidia-smi >/dev/null"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def _read_gpu_temp() -> int | None:
    sample = _read_gpu_sample()
    return None if sample is None else sample["temperature_c"]


def _read_gpu_sample() -> dict[str, int | float | None] | None:
    proc = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=temperature.gpu,power.draw,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
            "-i",
            "0",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if proc.returncode != 0:
        return None
    first = proc.stdout.strip().splitlines()[0] if proc.stdout.strip() else ""
    fields = [value.strip() for value in first.split(",")]
    if len(fields) != 4:
        return None

    def optional_float(value: str) -> float | None:
        try:
            return float(value)
        except ValueError:
            return None

    def optional_int(value: str) -> int | None:
        parsed = optional_float(value)
        return None if parsed is None else int(parsed)

    try:
        temperature = int(fields[0])
    except ValueError:
        return None
    return {
        "temperature_c": temperature,
        "power_w": optional_float(fields[1]),
        "memory_used_mib": optional_int(fields[2]),
        "utilization_pct": optional_int(fields[3]),
    }
