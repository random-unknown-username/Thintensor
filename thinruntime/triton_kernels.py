"""Triton decode kernels for ThinTensor's single-token runtime path."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import torch
import triton
import triton.language as tl


@triton.jit
def _single_token_gqa_attention_kernel(
    query,
    keys,
    values,
    output,
    tokens,
    stride_kk,
    stride_kt,
    stride_kd,
    stride_vk,
    stride_vt,
    stride_vd,
    heads: tl.constexpr,
    kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    scale: tl.constexpr,
    token_bucket: tl.constexpr,
    block_t: tl.constexpr,
    block_d: tl.constexpr,
) -> None:
    head = tl.program_id(0)
    kv_head = head // (heads // kv_heads)
    offsets_d = tl.arange(0, block_d)
    mask_d = offsets_d < head_dim
    q = tl.load(
        query + head * head_dim + offsets_d,
        mask=mask_d,
        other=0.0,
    ).to(tl.float32)

    running_max = -3.4028234663852886e38
    running_den = 0.0
    accumulator = tl.zeros((block_d,), dtype=tl.float32)

    for token_start in range(0, token_bucket, block_t):
        offsets_t = token_start + tl.arange(0, block_t)
        mask_t = offsets_t < tokens
        
        # Load keys
        key = tl.load(
            keys
            + kv_head * stride_kk
            + offsets_t[:, None] * stride_kt
            + offsets_d[None, :] * stride_kd,
            mask=mask_t[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.float32)
        
        scores = tl.sum(key * q[None, :], axis=1) * scale
        scores = tl.where(mask_t, scores, -3.4028234663852886e38)
        
        local_max = tl.max(scores, axis=0)
        new_max = tl.maximum(running_max, local_max)
        
        # Scale factor for historical terms
        scale_old = tl.exp(running_max - new_max)
        scale_old = tl.where(running_max == -3.4028234663852886e38, 0.0, scale_old)
        
        # Scale factor for current block terms
        scale_local = tl.exp(scores - new_max)
        scale_local = tl.where(mask_t, scale_local, 0.0)
        
        # Update running denominator
        running_den = running_den * scale_old + tl.sum(scale_local, axis=0)
        
        # Load values
        value = tl.load(
            values
            + kv_head * stride_vk
            + offsets_t[:, None] * stride_vt
            + offsets_d[None, :] * stride_vd,
            mask=mask_t[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.float32)
        
        # Update running accumulator
        accumulator = accumulator * scale_old + tl.sum(scale_local[:, None] * value, axis=0)
        running_max = new_max

    tl.store(
        output + head * head_dim + offsets_d,
        accumulator / running_den,
        mask=mask_d,
    )


@triton.jit
def _split_gqa_attention_partial_kernel(
    query,
    keys,
    values,
    partial_max,
    partial_den,
    partial_acc,
    tokens,
    stride_kk,
    stride_kt,
    stride_kd,
    stride_vk,
    stride_vt,
    stride_vd,
    heads: tl.constexpr,
    kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    scale: tl.constexpr,
    token_bucket: tl.constexpr,
    SPLITS: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
) -> None:
    head = tl.program_id(0)
    split = tl.program_id(1)
    kv_head = head // (heads // kv_heads)
    offsets_d = tl.arange(0, BLOCK_D)
    mask_d = offsets_d < head_dim
    query_values = tl.load(
        query + head * head_dim + offsets_d,
        mask=mask_d,
        other=0.0,
    ).to(tl.float32)
    chunk_tokens: tl.constexpr = token_bucket // SPLITS
    chunk_start = split * chunk_tokens

    running_max = -3.4028234663852886e38
    running_den = 0.0
    accumulator = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for local_start in range(0, chunk_tokens, BLOCK_T):
        offsets_t = chunk_start + local_start + tl.arange(0, BLOCK_T)
        mask_t = offsets_t < tokens
        key = tl.load(
            keys
            + kv_head * stride_kk
            + offsets_t[:, None] * stride_kt
            + offsets_d[None, :] * stride_kd,
            mask=mask_t[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.float32)
        scores = tl.sum(
            key * query_values[None, :],
            axis=1,
        ) * scale
        scores = tl.where(mask_t, scores, -3.4028234663852886e38)
        local_max = tl.max(scores, axis=0)
        next_max = tl.maximum(running_max, local_max)
        old_scale = tl.exp(running_max - next_max)
        old_scale = tl.where(
            running_max == -3.4028234663852886e38,
            0.0,
            old_scale,
        )
        probabilities = tl.exp(scores - next_max)
        probabilities = tl.where(mask_t, probabilities, 0.0)
        running_den = (
            running_den * old_scale
            + tl.sum(probabilities, axis=0)
        )
        value = tl.load(
            values
            + kv_head * stride_vk
            + offsets_t[:, None] * stride_vt
            + offsets_d[None, :] * stride_vd,
            mask=mask_t[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.float32)
        accumulator = (
            accumulator * old_scale
            + tl.sum(probabilities[:, None] * value, axis=0)
        )
        running_max = next_max

    partial_index = head * SPLITS + split
    tl.store(partial_max + partial_index, running_max)
    tl.store(partial_den + partial_index, running_den)
    tl.store(
        partial_acc + partial_index * head_dim + offsets_d,
        accumulator,
        mask=mask_d,
    )


@triton.jit
def _split_gqa_attention_reduce_kernel(
    partial_max,
    partial_den,
    partial_acc,
    output,
    heads: tl.constexpr,
    head_dim: tl.constexpr,
    SPLITS: tl.constexpr,
    BLOCK_D: tl.constexpr,
) -> None:
    head = tl.program_id(0)
    offsets_s = tl.arange(0, SPLITS)
    offsets_d = tl.arange(0, BLOCK_D)
    mask_d = offsets_d < head_dim
    max_values = tl.load(
        partial_max + head * SPLITS + offsets_s,
    )
    global_max = tl.max(max_values, axis=0)
    correction = tl.exp(max_values - global_max)
    den_values = tl.load(
        partial_den + head * SPLITS + offsets_s,
    )
    denominator = tl.sum(den_values * correction, axis=0)
    partial_values = tl.load(
        partial_acc
        + (head * SPLITS + offsets_s[:, None]) * head_dim
        + offsets_d[None, :],
        mask=mask_d[None, :],
        other=0.0,
    )
    accumulator = tl.sum(
        partial_values * correction[:, None],
        axis=0,
    )
    tl.store(
        output + head * head_dim + offsets_d,
        accumulator / denominator,
        mask=mask_d,
    )


@triton.jit
def _mxfp4_value(code):
    magnitude = code & 7
    value = tl.where(
        magnitude == 0,
        0.0,
        tl.where(
            magnitude == 1,
            0.5,
            tl.where(
                magnitude == 2,
                1.0,
                tl.where(
                    magnitude == 3,
                    1.5,
                    tl.where(
                        magnitude == 4,
                        2.0,
                        tl.where(
                            magnitude == 5,
                            3.0,
                            tl.where(magnitude == 6, 4.0, 6.0),
                        ),
                    ),
                ),
            ),
        ),
    )
    return tl.where((code & 8) != 0, -value, value)


@triton.jit
def _block_hadamard_32_kernel(x, out, cols: tl.constexpr) -> None:
    """Apply an orthonormal Walsh-Hadamard transform to each 32-value block."""
    block = tl.program_id(0)
    row = tl.arange(0, 32)
    col = tl.arange(0, 32)
    base = block * 32
    intersection = row[:, None] & col[None, :]
    parity = intersection ^ (intersection >> 16)
    parity = parity ^ (parity >> 8)
    parity = parity ^ (parity >> 4)
    parity = parity ^ (parity >> 2)
    parity = (parity ^ (parity >> 1)) & 1
    sign = 1.0 - 2.0 * parity.to(tl.float32)
    values = tl.load(
        x + base + col,
        mask=base + col < cols,
        other=0.0,
    ).to(tl.float32)
    transformed = tl.sum(sign * values[None, :], axis=1) * 0.1767766952966369
    tl.store(
        out + base + row,
        transformed,
        mask=base + row < cols,
    )


@triton.jit
def _block_hadamard_kernel(
    x,
    signs,
    out,
    cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    SIGNED: tl.constexpr,
) -> None:
    """Single-program fast Walsh-Hadamard transform for power-of-two blocks."""
    block = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    values = tl.load(x + block * BLOCK_SIZE + offsets).to(tl.float32)
    if SIGNED:
        values *= tl.load(
            signs + block * BLOCK_SIZE + offsets
        ).to(tl.float32)

    reshaped = tl.reshape(values, (BLOCK_SIZE // 2, 2, 1))
    reshaped = tl.permute(reshaped, (0, 2, 1))
    left, right = tl.split(reshaped)
    values = tl.reshape(
        tl.permute(tl.join(left + right, left - right), (0, 2, 1)),
        (BLOCK_SIZE,),
    )
    if BLOCK_SIZE >= 4:
        reshaped = tl.reshape(values, (BLOCK_SIZE // 4, 2, 2))
        reshaped = tl.permute(reshaped, (0, 2, 1))
        left, right = tl.split(reshaped)
        values = tl.reshape(
            tl.permute(tl.join(left + right, left - right), (0, 2, 1)),
            (BLOCK_SIZE,),
        )
    if BLOCK_SIZE >= 8:
        reshaped = tl.reshape(values, (BLOCK_SIZE // 8, 2, 4))
        reshaped = tl.permute(reshaped, (0, 2, 1))
        left, right = tl.split(reshaped)
        values = tl.reshape(
            tl.permute(tl.join(left + right, left - right), (0, 2, 1)),
            (BLOCK_SIZE,),
        )
    if BLOCK_SIZE >= 16:
        reshaped = tl.reshape(values, (BLOCK_SIZE // 16, 2, 8))
        reshaped = tl.permute(reshaped, (0, 2, 1))
        left, right = tl.split(reshaped)
        values = tl.reshape(
            tl.permute(tl.join(left + right, left - right), (0, 2, 1)),
            (BLOCK_SIZE,),
        )
    if BLOCK_SIZE >= 32:
        reshaped = tl.reshape(values, (BLOCK_SIZE // 32, 2, 16))
        reshaped = tl.permute(reshaped, (0, 2, 1))
        left, right = tl.split(reshaped)
        values = tl.reshape(
            tl.permute(tl.join(left + right, left - right), (0, 2, 1)),
            (BLOCK_SIZE,),
        )
    if BLOCK_SIZE >= 64:
        reshaped = tl.reshape(values, (BLOCK_SIZE // 64, 2, 32))
        reshaped = tl.permute(reshaped, (0, 2, 1))
        left, right = tl.split(reshaped)
        values = tl.reshape(
            tl.permute(tl.join(left + right, left - right), (0, 2, 1)),
            (BLOCK_SIZE,),
        )
    if BLOCK_SIZE >= 128:
        reshaped = tl.reshape(values, (BLOCK_SIZE // 128, 2, 64))
        reshaped = tl.permute(reshaped, (0, 2, 1))
        left, right = tl.split(reshaped)
        values = tl.reshape(
            tl.permute(tl.join(left + right, left - right), (0, 2, 1)),
            (BLOCK_SIZE,),
        )
    if BLOCK_SIZE >= 256:
        reshaped = tl.reshape(values, (BLOCK_SIZE // 256, 2, 128))
        reshaped = tl.permute(reshaped, (0, 2, 1))
        left, right = tl.split(reshaped)
        values = tl.reshape(
            tl.permute(tl.join(left + right, left - right), (0, 2, 1)),
            (BLOCK_SIZE,),
        )
    if BLOCK_SIZE >= 512:
        reshaped = tl.reshape(values, (BLOCK_SIZE // 512, 2, 256))
        reshaped = tl.permute(reshaped, (0, 2, 1))
        left, right = tl.split(reshaped)
        values = tl.reshape(
            tl.permute(tl.join(left + right, left - right), (0, 2, 1)),
            (BLOCK_SIZE,),
        )
    if BLOCK_SIZE >= 1024:
        reshaped = tl.reshape(values, (BLOCK_SIZE // 1024, 2, 512))
        reshaped = tl.permute(reshaped, (0, 2, 1))
        left, right = tl.split(reshaped)
        values = tl.reshape(
            tl.permute(tl.join(left + right, left - right), (0, 2, 1)),
            (BLOCK_SIZE,),
        )
    if BLOCK_SIZE >= 2048:
        reshaped = tl.reshape(values, (BLOCK_SIZE // 2048, 2, 1024))
        reshaped = tl.permute(reshaped, (0, 2, 1))
        left, right = tl.split(reshaped)
        values = tl.reshape(
            tl.permute(tl.join(left + right, left - right), (0, 2, 1)),
            (BLOCK_SIZE,),
        )
    values *= BLOCK_SIZE**-0.5
    tl.store(out + block * BLOCK_SIZE + offsets, values)


@triton.jit
def _mxfp4_selected_matvec_kernel(
    blocks,
    scales,
    x,
    expert_indices,
    out,
    rows: tl.constexpr,
    cols: tl.constexpr,
    stride_be: tl.constexpr,
    stride_br: tl.constexpr,
    stride_bg: tl.constexpr,
    stride_bb: tl.constexpr,
    stride_se: tl.constexpr,
    stride_sr: tl.constexpr,
    stride_sg: tl.constexpr,
    stride_xe: tl.constexpr,
    stride_oe: tl.constexpr,
    stride_or: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> None:
    row_pid = tl.program_id(0)
    selected_slot = tl.program_id(1)
    expert = tl.load(expert_indices + selected_slot)
    rows_offset = row_pid * BLOCK_M + tl.arange(0, BLOCK_M)
    cols_offset = tl.arange(0, BLOCK_N)
    row_mask = rows_offset < rows
    col_mask = cols_offset < cols
    group = cols_offset // 32
    byte_in_group = (cols_offset % 32) // 2
    packed = tl.load(
        blocks
        + expert * stride_be
        + rows_offset[:, None] * stride_br
        + group[None, :] * stride_bg
        + byte_in_group[None, :] * stride_bb,
        mask=row_mask[:, None] & col_mask[None, :],
        other=0,
    )
    nibble = tl.where(
        (cols_offset[None, :] & 1) == 0,
        packed & 0x0F,
        packed >> 4,
    )
    scale = tl.load(
        scales
        + expert * stride_se
        + rows_offset[:, None] * stride_sr
        + group[None, :] * stride_sg,
        mask=row_mask[:, None] & col_mask[None, :],
        other=127,
    ).to(tl.int32)
    weight = _mxfp4_value(nibble).to(tl.float32) * tl.exp2(
        (scale - 127).to(tl.float32)
    )
    vector = tl.load(
        x + selected_slot * stride_xe + cols_offset,
        mask=col_mask,
        other=0.0,
    ).to(tl.float32)
    result = tl.sum(weight * vector[None, :], axis=1)
    tl.store(
        out + selected_slot * stride_oe + rows_offset * stride_or,
        result,
        mask=row_mask,
    )


@triton.jit
def _matvec_kernel(
    weight,
    x,
    y,
    rows: tl.constexpr,
    cols: tl.constexpr,
    stride_wm: tl.constexpr,
    stride_wn: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> None:
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    mask_m = offs_m < rows
    mask_n = offs_n < cols

    w = tl.load(
        weight + offs_m[:, None] * stride_wm + offs_n[None, :] * stride_wn,
        mask=mask_m[:, None] & mask_n[None, :],
        other=0.0,
    ).to(tl.float32)
    xv = tl.load(x + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    acc = tl.sum(w * xv[None, :], axis=1)
    tl.store(y + offs_m, acc, mask=mask_m)


@triton.jit
def _matvec_nomask_n_kernel(
    weight,
    x,
    y,
    rows: tl.constexpr,
    cols: tl.constexpr,
    stride_wm: tl.constexpr,
    stride_wn: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> None:
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    mask_m = offs_m < rows
    w = tl.load(
        weight + offs_m[:, None] * stride_wm + offs_n[None, :] * stride_wn,
        mask=mask_m[:, None],
        other=0.0,
    ).to(tl.float32)
    xv = tl.load(x + offs_n).to(tl.float32)
    acc = tl.sum(w * xv[None, :], axis=1)
    tl.store(y + offs_m, acc, mask=mask_m)


@triton.jit
def _scaled_matvec_kernel(
    weight,
    scales,
    x,
    y,
    rows: tl.constexpr,
    cols: tl.constexpr,
    stride_wm: tl.constexpr,
    stride_wn: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> None:
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    mask_m = offs_m < rows
    mask_n = offs_n < cols
    w = tl.load(
        weight + offs_m[:, None] * stride_wm + offs_n[None, :] * stride_wn,
        mask=mask_m[:, None] & mask_n[None, :],
        other=0.0,
    ).to(tl.float32)
    xv = tl.load(x + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    scale = tl.load(scales + offs_m, mask=mask_m, other=0.0).to(tl.float32)
    acc = tl.sum(w * xv[None, :], axis=1) * scale
    tl.store(y + offs_m, acc, mask=mask_m)


@triton.jit
def _scaled_matvec_nomask_n_kernel(
    weight,
    scales,
    x,
    y,
    rows: tl.constexpr,
    cols: tl.constexpr,
    stride_wm: tl.constexpr,
    stride_wn: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> None:
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    mask_m = offs_m < rows
    w = tl.load(
        weight + offs_m[:, None] * stride_wm + offs_n[None, :] * stride_wn,
        mask=mask_m[:, None],
        other=0.0,
    ).to(tl.float32)
    xv = tl.load(x + offs_n).to(tl.float32)
    scale = tl.load(scales + offs_m, mask=mask_m, other=0.0).to(tl.float32)
    acc = tl.sum(w * xv[None, :], axis=1) * scale
    tl.store(y + offs_m, acc, mask=mask_m)


@triton.jit
def _fp8_bf16_tensorcore_matvec_kernel(
    weight,
    scales,
    x,
    y,
    rows: tl.constexpr,
    cols: tl.constexpr,
    stride_wm: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> None:
    offsets_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offsets_m < rows
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for start_k in range(0, cols, BLOCK_K):
        offsets_k = start_k + tl.arange(0, BLOCK_K)
        weight_values = tl.load(
            weight
            + offsets_m[:, None] * stride_wm
            + offsets_k[None, :],
            mask=mask_m[:, None],
            other=0.0,
        )
        vector = tl.load(x + offsets_k)
        vector_tile = vector[:, None] + tl.zeros(
            (BLOCK_K, BLOCK_N),
            dtype=tl.bfloat16,
        )
        accumulator = tl.dot_scaled(
            weight_values,
            None,
            "e4m3",
            vector_tile,
            None,
            "bf16",
            acc=accumulator,
            fast_math=True,
        )
    scale = tl.load(scales + offsets_m, mask=mask_m, other=0.0)
    result = tl.sum(accumulator, axis=1) / BLOCK_N
    tl.store(y + offsets_m, result * scale, mask=mask_m)


@triton.jit
def _int8_bf16_tensorcore_matvec_kernel(
    weight,
    scales,
    x,
    y,
    rows: tl.constexpr,
    cols: tl.constexpr,
    stride_wm: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> None:
    offsets_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offsets_m < rows
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for start_k in range(0, cols, BLOCK_K):
        offsets_k = start_k + tl.arange(0, BLOCK_K)
        weight_values = tl.load(
            weight
            + offsets_m[:, None] * stride_wm
            + offsets_k[None, :],
            mask=mask_m[:, None],
            other=0,
        ).to(tl.bfloat16)
        vector = tl.load(x + offsets_k)
        vector_tile = vector[:, None] + tl.zeros(
            (BLOCK_K, BLOCK_N),
            dtype=tl.bfloat16,
        )
        accumulator += tl.dot(weight_values, vector_tile)
    scale = tl.load(scales + offsets_m, mask=mask_m, other=0.0)
    result = tl.sum(accumulator, axis=1) / BLOCK_N
    tl.store(y + offsets_m, result * scale, mask=mask_m)


@triton.jit
def _mxfp4_bf16_tensorcore_matvec_kernel(
    packed_weight,
    scales,
    residual_bits,
    residual_scales,
    post_scales,
    x,
    y,
    rows: tl.constexpr,
    cols: tl.constexpr,
    stride_wm: tl.constexpr,
    stride_sm: tl.constexpr,
    stride_sk: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_BINARY_RESIDUAL: tl.constexpr,
    HAS_POST_SCALE: tl.constexpr,
) -> None:
    """Blackwell MXFP4 x BF16 matvec using native scaled tensor cores."""
    offsets_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offsets_m < rows
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for start_k in range(0, cols, BLOCK_K):
        offsets_packed_k = start_k // 2 + tl.arange(0, BLOCK_K // 2)
        weight_values = tl.load(
            packed_weight
            + offsets_m[:, None] * stride_wm
            + offsets_packed_k[None, :],
            mask=mask_m[:, None],
            other=0,
        )
        offsets_scale_k = (
            start_k // 32 + tl.arange(0, BLOCK_K // 32)
        )
        weight_scales = tl.load(
            scales
            + offsets_m[:, None] * stride_sm
            + offsets_scale_k[None, :] * stride_sk,
            mask=mask_m[:, None],
            other=0.0,
        )
        offsets_k = start_k + tl.arange(0, BLOCK_K)
        vector = tl.load(x + offsets_k)
        vector_tile = vector[:, None] + tl.zeros(
            (BLOCK_K, BLOCK_N),
            dtype=tl.bfloat16,
        )
        accumulator = tl.dot_scaled(
            weight_values,
            weight_scales,
            "e2m1",
            vector_tile,
            None,
            "bf16",
            acc=accumulator,
            fast_math=True,
        )
        if HAS_BINARY_RESIDUAL:
            pair_index = start_k // 2 + tl.arange(0, BLOCK_K // 2)
            low_index = pair_index * 2
            high_index = low_index + 1
            low_bits = tl.load(
                residual_bits
                + offsets_m[:, None] * (cols // 8)
                + (low_index[None, :] // 8),
                mask=mask_m[:, None],
                other=0,
            )
            high_bits = tl.load(
                residual_bits
                + offsets_m[:, None] * (cols // 8)
                + (high_index[None, :] // 8),
                mask=mask_m[:, None],
                other=0,
            )
            low_positive = (
                (low_bits >> (low_index[None, :] & 7)) & 1
            ) != 0
            high_positive = (
                (high_bits >> (high_index[None, :] & 7)) & 1
            ) != 0
            low_code = tl.where(low_positive, 2, 10).to(tl.uint8)
            high_code = tl.where(high_positive, 2, 10).to(tl.uint8)
            residual_values = low_code | (high_code << 4)
            residual_scale_values = tl.load(
                residual_scales
                + offsets_m[:, None] * (cols // 32)
                + offsets_scale_k[None, :],
                mask=mask_m[:, None],
                other=127,
            )
            accumulator = tl.dot_scaled(
                residual_values,
                residual_scale_values,
                "e2m1",
                vector_tile,
                None,
                "bf16",
                acc=accumulator,
                fast_math=True,
            )
    result = tl.sum(accumulator, axis=1) / BLOCK_N
    if HAS_POST_SCALE:
        result *= tl.load(post_scales + offsets_m, mask=mask_m, other=1.0)
    tl.store(y + offsets_m, result, mask=mask_m)


@triton.jit
def _int4_scaled_matvec_kernel(
    packed_weight,
    scales,
    x,
    y,
    rows: tl.constexpr,
    cols: tl.constexpr,
    stride_wm: tl.constexpr,
    stride_sm: tl.constexpr,
    stride_sn: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> None:
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_b = tl.arange(0, BLOCK_N)
    packed_cols: tl.constexpr = (cols + 1) // 2
    mask_m = offs_m < rows
    mask_b = offs_b < packed_cols
    packed = tl.load(
        packed_weight
        + offs_m[:, None] * stride_wm
        + offs_b[None, :],
        mask=mask_m[:, None] & mask_b[None, :],
        other=0,
    ).to(tl.int32)
    quantized_low = (packed & 0x0F) - 8
    quantized_high = ((packed >> 4) & 0x0F) - 8
    if GROUP_SIZE >= cols:
        scale = tl.load(
            scales + offs_m * stride_sm,
            mask=mask_m,
            other=0.0,
        ).to(tl.float32)[:, None]
    else:
        scale = tl.load(
            scales
            + offs_m[:, None] * stride_sm
            + ((offs_b[None, :] * 2) // GROUP_SIZE) * stride_sn,
            mask=mask_m[:, None] & mask_b[None, :],
            other=0.0,
        ).to(tl.float32)
    vector_low = tl.load(
        x + offs_b * 2,
        mask=mask_b,
        other=0.0,
    ).to(tl.float32)
    vector_high = tl.load(
        x + offs_b * 2 + 1,
        mask=mask_b & (offs_b * 2 + 1 < cols),
        other=0.0,
    ).to(tl.float32)
    accumulator = tl.sum(
        (
            quantized_low.to(tl.float32) * vector_low[None, :]
            + quantized_high.to(tl.float32) * vector_high[None, :]
        )
        * scale,
        axis=1,
    )
    tl.store(y + offs_m, accumulator, mask=mask_m)


@triton.jit
def _int4_scaled_matvec_looped_kernel(
    packed_weight,
    scales,
    x,
    y,
    rows: tl.constexpr,
    cols: tl.constexpr,
    stride_wm: tl.constexpr,
    stride_sm: tl.constexpr,
    stride_sn: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> None:
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < rows
    accumulator = tl.zeros((BLOCK_M,), dtype=tl.float32)
    packed_cols: tl.constexpr = (cols + 1) // 2
    for start_b in range(0, packed_cols, BLOCK_N):
        offs_b = start_b + tl.arange(0, BLOCK_N)
        mask_b = offs_b < packed_cols
        packed = tl.load(
            packed_weight
            + offs_m[:, None] * stride_wm
            + offs_b[None, :],
            mask=mask_m[:, None] & mask_b[None, :],
            other=0,
        ).to(tl.int32)
        quantized_low = (packed & 0x0F) - 8
        quantized_high = ((packed >> 4) & 0x0F) - 8
        if GROUP_SIZE >= cols:
            scale = tl.load(
                scales + offs_m * stride_sm,
                mask=mask_m,
                other=0.0,
            ).to(tl.float32)[:, None]
        else:
            scale = tl.load(
                scales
                + offs_m[:, None] * stride_sm
                + ((offs_b[None, :] * 2) // GROUP_SIZE) * stride_sn,
                mask=mask_m[:, None] & mask_b[None, :],
                other=0.0,
            ).to(tl.float32)
        vector_low = tl.load(
            x + offs_b * 2,
            mask=mask_b,
            other=0.0,
        ).to(tl.float32)
        vector_high = tl.load(
            x + offs_b * 2 + 1,
            mask=mask_b & (offs_b * 2 + 1 < cols),
            other=0.0,
        ).to(tl.float32)
        accumulator += tl.sum(
            (
                quantized_low.to(tl.float32) * vector_low[None, :]
                + quantized_high.to(tl.float32) * vector_high[None, :]
            )
            * scale,
            axis=1,
        )
    tl.store(y + offs_m, accumulator, mask=mask_m)


@triton.jit
def _int4_dot_matvec_kernel(
    packed_weight,
    scales,
    x,
    y,
    rows: tl.constexpr,
    cols: tl.constexpr,
    stride_wm: tl.constexpr,
    stride_sm: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_OUT: tl.constexpr,
) -> None:
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < rows
    offs_out = tl.arange(0, BLOCK_OUT)
    accumulator = tl.zeros(
        (BLOCK_M, BLOCK_OUT),
        dtype=tl.float32,
    )
    for start_k in range(0, cols, BLOCK_K):
        offs_k = start_k + tl.arange(0, BLOCK_K)
        mask_k = offs_k < cols
        packed = tl.load(
            packed_weight
            + offs_m[:, None] * stride_wm
            + (offs_k[None, :] // 2),
            mask=mask_m[:, None] & mask_k[None, :],
            other=0,
        ).to(tl.int32)
        shift = (offs_k[None, :] & 1) * 4
        quantized = ((packed >> shift) & 0x0F) - 8
        vector = tl.load(
            x + offs_k,
            mask=mask_k,
            other=0.0,
        ).to(tl.bfloat16)
        vector_tile = vector[:, None] + tl.zeros(
            (BLOCK_K, BLOCK_OUT),
            dtype=tl.bfloat16,
        )
        accumulator += tl.dot(
            quantized.to(tl.bfloat16),
            vector_tile,
        )
    scale = tl.load(
        scales + offs_m * stride_sm,
        mask=mask_m,
        other=0.0,
    ).to(tl.float32)
    tl.store(
        y + offs_m,
        (tl.sum(accumulator, axis=1) / BLOCK_OUT) * scale,
        mask=mask_m,
    )


@triton.jit
def _multi_int4_scaled_matvec_kernel(
    weight0,
    weight1,
    weight2,
    scale0,
    scale1,
    scale2,
    x,
    out,
    rows0: tl.constexpr,
    rows1: tl.constexpr,
    rows2: tl.constexpr,
    cols: tl.constexpr,
    stride_w0: tl.constexpr,
    stride_w1: tl.constexpr,
    stride_w2: tl.constexpr,
    stride_s0: tl.constexpr,
    stride_s1: tl.constexpr,
    stride_s2: tl.constexpr,
    BLOCKS0: tl.constexpr,
    BLOCKS1: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_B: tl.constexpr,
) -> None:
    pid = tl.program_id(0)
    in_matrix1 = pid >= BLOCKS0
    in_matrix2 = pid >= BLOCKS0 + BLOCKS1
    local_pid = tl.where(
        in_matrix2,
        pid - BLOCKS0 - BLOCKS1,
        tl.where(in_matrix1, pid - BLOCKS0, pid),
    )
    rows = tl.where(in_matrix2, rows2, tl.where(in_matrix1, rows1, rows0))
    row_base = tl.where(
        in_matrix2,
        rows0 + rows1,
        tl.where(in_matrix1, rows0, 0),
    )
    weight = tl.where(
        in_matrix2,
        weight2,
        tl.where(in_matrix1, weight1, weight0),
    )
    scale_ptr = tl.where(
        in_matrix2,
        scale2,
        tl.where(in_matrix1, scale1, scale0),
    )
    stride_w = tl.where(
        in_matrix2,
        stride_w2,
        tl.where(in_matrix1, stride_w1, stride_w0),
    )
    stride_s = tl.where(
        in_matrix2,
        stride_s2,
        tl.where(in_matrix1, stride_s1, stride_s0),
    )
    offs_m = local_pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_b = tl.arange(0, BLOCK_B)
    packed_cols: tl.constexpr = (cols + 1) // 2
    mask_m = offs_m < rows
    mask_b = offs_b < packed_cols
    packed = tl.load(
        weight + offs_m[:, None] * stride_w + offs_b[None, :],
        mask=mask_m[:, None] & mask_b[None, :],
        other=0,
    ).to(tl.int32)
    quantized_low = (packed & 0x0F) - 8
    quantized_high = ((packed >> 4) & 0x0F) - 8
    vector_low = tl.load(
        x + offs_b * 2,
        mask=mask_b,
        other=0.0,
    ).to(tl.float32)
    vector_high = tl.load(
        x + offs_b * 2 + 1,
        mask=mask_b & (offs_b * 2 + 1 < cols),
        other=0.0,
    ).to(tl.float32)
    scale = tl.load(
        scale_ptr + offs_m * stride_s,
        mask=mask_m,
        other=0.0,
    ).to(tl.float32)
    accumulator = tl.sum(
        quantized_low.to(tl.float32) * vector_low[None, :]
        + quantized_high.to(tl.float32) * vector_high[None, :],
        axis=1,
    )
    tl.store(
        out + row_base + offs_m,
        accumulator * scale,
        mask=mask_m,
    )


@triton.jit
def _multi_scaled_tensorcore_matvec_kernel(
    weight0,
    weight1,
    weight2,
    scale0,
    scale1,
    scale2,
    x,
    out,
    rows0: tl.constexpr,
    rows1: tl.constexpr,
    rows2: tl.constexpr,
    cols: tl.constexpr,
    stride_w0: tl.constexpr,
    stride_w1: tl.constexpr,
    stride_w2: tl.constexpr,
    stride_s0: tl.constexpr,
    stride_s1: tl.constexpr,
    stride_s2: tl.constexpr,
    BLOCKS0: tl.constexpr,
    BLOCKS1: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_FP8: tl.constexpr,
) -> None:
    pid = tl.program_id(0)
    in_matrix1 = pid >= BLOCKS0
    in_matrix2 = pid >= BLOCKS0 + BLOCKS1
    local_pid = tl.where(
        in_matrix2,
        pid - BLOCKS0 - BLOCKS1,
        tl.where(in_matrix1, pid - BLOCKS0, pid),
    )
    rows = tl.where(in_matrix2, rows2, tl.where(in_matrix1, rows1, rows0))
    row_base = tl.where(
        in_matrix2,
        rows0 + rows1,
        tl.where(in_matrix1, rows0, 0),
    )
    weight = tl.where(
        in_matrix2,
        weight2,
        tl.where(in_matrix1, weight1, weight0),
    )
    scale_ptr = tl.where(
        in_matrix2,
        scale2,
        tl.where(in_matrix1, scale1, scale0),
    )
    stride_w = tl.where(
        in_matrix2,
        stride_w2,
        tl.where(in_matrix1, stride_w1, stride_w0),
    )
    stride_s = tl.where(
        in_matrix2,
        stride_s2,
        tl.where(in_matrix1, stride_s1, stride_s0),
    )
    offsets_m = local_pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offsets_m < rows
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for start_k in range(0, cols, BLOCK_K):
        offsets_k = start_k + tl.arange(0, BLOCK_K)
        weight_pointer = (
            weight
            + offsets_m[:, None] * stride_w
            + offsets_k[None, :]
        )
        if IS_FP8:
            weight_values = tl.load(
                weight_pointer,
                mask=mask_m[:, None],
                other=0.0,
            )
        else:
            weight_values = tl.load(
                weight_pointer,
                mask=mask_m[:, None],
                other=0,
            )
        vector = tl.load(x + offsets_k)
        vector_tile = vector[:, None] + tl.zeros(
            (BLOCK_K, BLOCK_N),
            dtype=tl.bfloat16,
        )
        if IS_FP8:
            accumulator = tl.dot_scaled(
                weight_values,
                None,
                "e4m3",
                vector_tile,
                None,
                "bf16",
                acc=accumulator,
                fast_math=True,
            )
        else:
            accumulator += tl.dot(
                weight_values.to(tl.bfloat16),
                vector_tile,
            )
    scale = tl.load(
        scale_ptr + offsets_m * stride_s,
        mask=mask_m,
        other=0.0,
    )
    result = tl.sum(accumulator, axis=1) / BLOCK_N
    tl.store(
        out + row_base + offsets_m,
        result * scale,
        mask=mask_m,
    )


@triton.jit
def _multi_matvec_nomask_n_kernel(
    weight0,
    weight1,
    weight2,
    x,
    out,
    rows0: tl.constexpr,
    rows1: tl.constexpr,
    rows2: tl.constexpr,
    cols: tl.constexpr,
    stride0_m: tl.constexpr,
    stride1_m: tl.constexpr,
    stride2_m: tl.constexpr,
    BLOCKS0: tl.constexpr,
    BLOCKS1: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> None:
    pid = tl.program_id(0)
    in_matrix1 = pid >= BLOCKS0
    in_matrix2 = pid >= BLOCKS0 + BLOCKS1
    local_pid = tl.where(
        in_matrix2,
        pid - BLOCKS0 - BLOCKS1,
        tl.where(in_matrix1, pid - BLOCKS0, pid),
    )
    rows = tl.where(in_matrix2, rows2, tl.where(in_matrix1, rows1, rows0))
    row_base = tl.where(
        in_matrix2,
        rows0 + rows1,
        tl.where(in_matrix1, rows0, 0),
    )
    stride_m = tl.where(
        in_matrix2,
        stride2_m,
        tl.where(in_matrix1, stride1_m, stride0_m),
    )
    weight = tl.where(
        in_matrix2,
        weight2,
        tl.where(in_matrix1, weight1, weight0),
    )

    offs_m = local_pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    mask_m = offs_m < rows
    w = tl.load(
        weight + offs_m[:, None] * stride_m + offs_n[None, :],
        mask=mask_m[:, None],
        other=0.0,
    ).to(tl.float32)
    xv = tl.load(x + offs_n).to(tl.float32)
    acc = tl.sum(w * xv[None, :], axis=1)
    tl.store(out + row_base + offs_m, acc, mask=mask_m)


@triton.jit
def _repeat_kv_matvec_nomask_n_kernel(
    weight,
    value,
    out,
    rows: tl.constexpr,
    cols: tl.constexpr,
    stride_wm: tl.constexpr,
    stride_wn: tl.constexpr,
    heads: tl.constexpr,
    kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> None:
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    mask_m = offs_m < rows
    group = heads // kv_heads
    head = offs_n // head_dim
    dim = offs_n - head * head_dim
    value_offset = (head // group) * head_dim + dim
    w = tl.load(
        weight + offs_m[:, None] * stride_wm + offs_n[None, :] * stride_wn,
        mask=mask_m[:, None],
        other=0.0,
    ).to(tl.float32)
    xv = tl.load(value + value_offset).to(tl.float32)
    acc = tl.sum(w * xv[None, :], axis=1)
    tl.store(out + offs_m, acc, mask=mask_m)


@triton.jit
def _matvec_argmax_stage1_kernel(
    weight,
    x,
    partial_vals,
    partial_idxs,
    rows: tl.constexpr,
    cols: tl.constexpr,
    stride_wm: tl.constexpr,
    stride_wn: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> None:
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    mask_m = offs_m < rows
    mask_n = offs_n < cols
    w = tl.load(
        weight + offs_m[:, None] * stride_wm + offs_n[None, :] * stride_wn,
        mask=mask_m[:, None] & mask_n[None, :],
        other=0.0,
    ).to(tl.float32)
    xv = tl.load(x + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    acc = tl.sum(w * xv[None, :], axis=1)
    neg_inf = -3.4028234663852886e38
    vals = tl.where(mask_m, acc, neg_inf)
    max_val = tl.max(vals, axis=0)
    idx_candidates = tl.where(vals == max_val, offs_m, 0)
    max_idx = tl.max(idx_candidates, axis=0)
    tl.store(partial_vals + pid, max_val)
    tl.store(partial_idxs + pid, max_idx)


@triton.jit
def _matvec_argmax_stage1_nomask_n_kernel(
    weight,
    x,
    partial_vals,
    partial_idxs,
    rows: tl.constexpr,
    cols: tl.constexpr,
    stride_wm: tl.constexpr,
    stride_wn: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> None:
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    mask_m = offs_m < rows
    w = tl.load(
        weight + offs_m[:, None] * stride_wm + offs_n[None, :] * stride_wn,
        mask=mask_m[:, None],
        other=0.0,
    ).to(tl.float32)
    xv = tl.load(x + offs_n).to(tl.float32)
    acc = tl.sum(w * xv[None, :], axis=1)
    neg_inf = -3.4028234663852886e38
    vals = tl.where(mask_m, acc, neg_inf)
    max_val = tl.max(vals, axis=0)
    idx_candidates = tl.where(vals == max_val, offs_m, 0)
    max_idx = tl.max(idx_candidates, axis=0)
    tl.store(partial_vals + pid, max_val)
    tl.store(partial_idxs + pid, max_idx)


@triton.jit
def _indexed_matvec_argmax_kernel(
    weight,
    x,
    indices,
    out_idx,
    rows: tl.constexpr,
    cols: tl.constexpr,
    candidate_count: tl.constexpr,
    stride_wm: tl.constexpr,
    stride_wn: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> None:
    offs_k = tl.arange(0, BLOCK_K)
    offs_n = tl.arange(0, BLOCK_N)
    mask_k = offs_k < candidate_count
    mask_n = offs_n < cols
    token_ids = tl.load(
        indices + offs_k,
        mask=mask_k,
        other=rows,
    )
    weight_values = tl.load(
        weight
        + token_ids[:, None] * stride_wm
        + offs_n[None, :] * stride_wn,
        mask=mask_k[:, None] & mask_n[None, :],
        other=0.0,
    ).to(tl.float32)
    x_values = tl.load(
        x + offs_n,
        mask=mask_n,
        other=0.0,
    ).to(tl.float32)
    # The normal head path materializes BF16 logits before torch.argmax.
    logits = tl.sum(
        weight_values * x_values[None, :],
        axis=1,
    ).to(tl.bfloat16).to(tl.float32)
    logits = tl.where(mask_k, logits, -3.4028234663852886e38)
    max_logit = tl.max(logits, axis=0)
    tied_ids = tl.where(logits == max_logit, token_ids, rows)
    tl.store(out_idx, tl.min(tied_ids, axis=0))


@triton.jit
def _sparse_residual_matvec_kernel(
    residual_values,
    residual_indices,
    x,
    out,
    rows: tl.constexpr,
    terms: tl.constexpr,
    stride_vm: tl.constexpr,
    stride_vk: tl.constexpr,
    stride_im: tl.constexpr,
    stride_ik: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
) -> None:
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < rows
    mask_k = offs_k < terms
    indices = tl.load(
        residual_indices
        + offs_m[:, None] * stride_im
        + offs_k[None, :] * stride_ik,
        mask=mask_m[:, None] & mask_k[None, :],
        other=0,
    )
    values = tl.load(
        residual_values
        + offs_m[:, None] * stride_vm
        + offs_k[None, :] * stride_vk,
        mask=mask_m[:, None] & mask_k[None, :],
        other=0.0,
    ).to(tl.float32)
    x_values = tl.load(
        x + indices,
        mask=mask_m[:, None] & mask_k[None, :],
        other=0.0,
    ).to(tl.float32)
    correction = tl.sum(values * x_values, axis=1)
    base = tl.load(out + offs_m, mask=mask_m, other=0.0).to(tl.float32)
    tl.store(out + offs_m, base + correction, mask=mask_m)


@triton.jit
def _binary_residual_matvec_kernel(
    packed_signs,
    scales,
    x,
    out,
    rows: tl.constexpr,
    cols: tl.constexpr,
    stride_bm: tl.constexpr,
    stride_sm: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
) -> None:
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < rows
    correction = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for start_k in range(0, cols, BLOCK_K):
        offs_k = start_k + tl.arange(0, BLOCK_K)
        mask_k = offs_k < cols
        packed = tl.load(
            packed_signs
            + offs_m[:, None] * stride_bm
            + (offs_k[None, :] // 8),
            mask=mask_m[:, None] & mask_k[None, :],
            other=0,
        ).to(tl.int32)
        positive = ((packed >> (offs_k[None, :] & 7)) & 1) != 0
        sign = tl.where(positive, 1.0, -1.0)
        scale = tl.load(
            scales
            + offs_m[:, None] * stride_sm
            + (offs_k[None, :] // 32),
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        ).to(tl.float32)
        values = tl.load(
            x + offs_k,
            mask=mask_k,
            other=0.0,
        ).to(tl.float32)
        correction += tl.sum(sign * scale * values[None, :], axis=1)
    base = tl.load(out + offs_m, mask=mask_m, other=0.0).to(tl.float32)
    tl.store(out + offs_m, base + correction, mask=mask_m)


@triton.jit
def _selected_scaled_matvec_kernel(
    weight,
    scales,
    row_indices,
    x,
    out,
    rows: tl.constexpr,
    cols: tl.constexpr,
    stride_wm: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
) -> None:
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < rows
    accumulator = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for start_k in range(0, cols, BLOCK_K):
        offs_k = start_k + tl.arange(0, BLOCK_K)
        mask_k = offs_k < cols
        values = tl.load(
            weight
            + offs_m[:, None] * stride_wm
            + offs_k[None, :],
            mask=mask_m[:, None] & mask_k[None, :],
            other=0,
        ).to(tl.float32)
        vector = tl.load(x + offs_k, mask=mask_k, other=0).to(tl.float32)
        accumulator += tl.sum(values * vector[None, :], axis=1)
    scale = tl.load(scales + offs_m, mask=mask_m, other=0).to(tl.float32)
    destination = tl.load(row_indices + offs_m, mask=mask_m, other=0)
    tl.store(out + destination, accumulator * scale, mask=mask_m)


@triton.jit
def _argmax_stage2_kernel(
    partial_vals,
    partial_idxs,
    out_idx,
    out_val,
    n: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> None:
    offs = tl.arange(0, BLOCK_N)
    mask = offs < n
    neg_inf = -3.4028234663852886e38
    vals = tl.load(partial_vals + offs, mask=mask, other=neg_inf).to(tl.float32)
    idxs = tl.load(partial_idxs + offs, mask=mask, other=0).to(tl.int64)
    max_val = tl.max(vals, axis=0)
    idx_candidates = tl.where(vals == max_val, idxs, 0)
    max_idx = tl.max(idx_candidates, axis=0)
    tl.store(out_idx, max_idx)
    tl.store(out_val, max_val)


@triton.jit
def _persistent_vocab_block_argmax_kernel(
    weight,
    x,
    out_idx,
    out_val,
    counter,
    partial_vals,
    partial_idxs,
    rows: tl.constexpr,
    cols: tl.constexpr,
    stride_wm: tl.constexpr,
    stride_wn: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_PROGRAMS: tl.constexpr,
) -> None:
    pid = tl.program_id(0)

    neg_inf = -3.4028234663852886e38
    local_max_val = neg_inf
    local_max_idx = tl.zeros((), dtype=tl.int64) + rows

    step = NUM_PROGRAMS * BLOCK_M
    start_row = pid * BLOCK_M

    offs_n = tl.arange(0, BLOCK_N)
    mask_n = offs_n < cols
    xv = tl.load(x + offs_n, mask=mask_n, other=0.0).to(tl.float32)

    curr_row = start_row
    while curr_row < rows:
        offs_m = curr_row + tl.arange(0, BLOCK_M)
        mask_m = offs_m < rows

        w = tl.load(
            weight + offs_m[:, None] * stride_wm + offs_n[None, :] * stride_wn,
            mask=mask_m[:, None] & mask_n[None, :],
            other=0.0,
        ).to(tl.float32)

        acc = tl.sum(w * xv[None, :], axis=1)
        vals = tl.where(mask_m, acc, neg_inf)

        block_max_val = tl.max(vals, axis=0)
        tied_ids = tl.where(vals == block_max_val, offs_m, rows)
        block_min_idx = tl.min(tied_ids, axis=0)

        is_greater = block_max_val > local_max_val
        is_equal = block_max_val == local_max_val

        local_max_idx = tl.where(
            is_greater,
            block_min_idx,
            tl.where(
                is_equal,
                tl.minimum(local_max_idx, block_min_idx),
                local_max_idx,
            ),
        )
        local_max_val = tl.where(is_greater, block_max_val, local_max_val)

        curr_row += step

    tl.store(partial_vals + pid, local_max_val)
    tl.store(partial_idxs + pid, local_max_idx)

    oldval = tl.atomic_add(counter, 1)

    if oldval == NUM_PROGRAMS - 1:
        BLOCK_P: tl.constexpr = NUM_PROGRAMS
        offs_p = tl.arange(0, BLOCK_P)
        mask_p = offs_p < NUM_PROGRAMS

        p_vals = tl.load(partial_vals + offs_p, mask=mask_p, other=neg_inf).to(tl.float32)
        p_idxs = tl.load(partial_idxs + offs_p, mask=mask_p, other=rows).to(tl.int64)

        global_max_val = tl.max(p_vals, axis=0)
        tied_p_ids = tl.where(p_vals == global_max_val, p_idxs, rows)
        global_max_idx = tl.min(tied_p_ids, axis=0)

        tl.store(out_idx, global_max_idx)
        tl.store(out_val, global_max_val)


@triton.jit
def _rms_norm_kernel(
    x,
    weight,
    y,
    n: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> None:
    offs = tl.arange(0, BLOCK_N)
    mask = offs < n
    xv = tl.load(x + offs, mask=mask, other=0.0).to(tl.float32)
    wv = tl.load(weight + offs, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(xv * xv, axis=0) / n
    out = xv * tl.rsqrt(mean + eps) * wv
    tl.store(y + offs, out, mask=mask)


@triton.jit
def _add_rms_norm_kernel(
    left,
    right,
    residual_out,
    weight,
    norm_out,
    n: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> None:
    offs = tl.arange(0, BLOCK_N)
    mask = offs < n
    left_values = tl.load(
        left + offs,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    right_values = tl.load(
        right + offs,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    # Preserve the existing add-kernel BF16 store/reload boundary before RMS.
    residual = (left_values + right_values).to(tl.bfloat16)
    tl.store(residual_out + offs, residual, mask=mask)
    residual_fp32 = residual.to(tl.float32)
    weight_values = tl.load(
        weight + offs,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    mean = tl.sum(residual_fp32 * residual_fp32, axis=0) / n
    normalized = residual_fp32 * tl.rsqrt(mean + eps) * weight_values
    tl.store(norm_out + offs, normalized, mask=mask)


@triton.jit
def _head_rms_norm_kernel(
    x,
    weight,
    y,
    head_dim: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> None:
    head = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < head_dim
    base = head * head_dim
    xv = tl.load(x + base + offs, mask=mask, other=0.0).to(tl.float32)
    wv = tl.load(weight + offs, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(xv * xv, axis=0) / head_dim
    out = xv * tl.rsqrt(mean + eps) * wv
    tl.store(y + base + offs, out, mask=mask)


@triton.jit
def _rope_qk_inplace_kernel(
    q,
    k,
    cos,
    sin,
    heads: tl.constexpr,
    kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    HALF: tl.constexpr,
    BLOCK_HALF: tl.constexpr,
) -> None:
    head = tl.program_id(0)
    offs = tl.arange(0, BLOCK_HALF)
    mask = offs < HALF
    cos_values = tl.load(cos + offs, mask=mask, other=0.0).to(
        tl.float32
    )
    sin_values = tl.load(sin + offs, mask=mask, other=0.0).to(
        tl.float32
    )

    q_mask = mask & (head < heads)
    q_base = head * head_dim
    q_first = tl.load(
        q + q_base + offs,
        mask=q_mask,
        other=0.0,
    ).to(tl.float32)
    q_second = tl.load(
        q + q_base + HALF + offs,
        mask=q_mask,
        other=0.0,
    ).to(tl.float32)
    # PyTorch BF16 performs and rounds each multiply before add/subtract.
    q_first_cos = (q_first * cos_values).to(tl.bfloat16)
    q_second_sin = (q_second * sin_values).to(tl.bfloat16)
    q_second_cos = (q_second * cos_values).to(tl.bfloat16)
    q_first_sin = (q_first * sin_values).to(tl.bfloat16)
    q_out_first = (
        q_first_cos.to(tl.float32) - q_second_sin.to(tl.float32)
    ).to(tl.bfloat16)
    q_out_second = (
        q_second_cos.to(tl.float32) + q_first_sin.to(tl.float32)
    ).to(tl.bfloat16)
    tl.store(q + q_base + offs, q_out_first, mask=q_mask)
    tl.store(q + q_base + HALF + offs, q_out_second, mask=q_mask)

    k_mask = mask & (head < kv_heads)
    k_base = head * head_dim
    k_first = tl.load(
        k + k_base + offs,
        mask=k_mask,
        other=0.0,
    ).to(tl.float32)
    k_second = tl.load(
        k + k_base + HALF + offs,
        mask=k_mask,
        other=0.0,
    ).to(tl.float32)
    k_first_cos = (k_first * cos_values).to(tl.bfloat16)
    k_second_sin = (k_second * sin_values).to(tl.bfloat16)
    k_second_cos = (k_second * cos_values).to(tl.bfloat16)
    k_first_sin = (k_first * sin_values).to(tl.bfloat16)
    k_out_first = (
        k_first_cos.to(tl.float32) - k_second_sin.to(tl.float32)
    ).to(tl.bfloat16)
    k_out_second = (
        k_second_cos.to(tl.float32) + k_first_sin.to(tl.float32)
    ).to(tl.bfloat16)
    tl.store(k + k_base + offs, k_out_first, mask=k_mask)
    tl.store(k + k_base + HALF + offs, k_out_second, mask=k_mask)


@triton.jit
def _silu_mul_kernel(
    gate,
    up,
    out,
    n: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> None:
    pid = tl.program_id(0)
    offs = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < n
    gv = tl.load(gate + offs, mask=mask, other=0.0).to(tl.float32)
    uv = tl.load(up + offs, mask=mask, other=0.0).to(tl.float32)
    silu = gv / (1.0 + tl.exp(-gv))
    tl.store(out + offs, silu * uv, mask=mask)


@triton.jit
def _add_kernel(
    left,
    right,
    out,
    n: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> None:
    pid = tl.program_id(0)
    offs = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < n
    lv = tl.load(left + offs, mask=mask, other=0.0)
    rv = tl.load(right + offs, mask=mask, other=0.0)
    tl.store(out + offs, lv + rv, mask=mask)


@triton.jit
def _repeat_kv_kernel(
    v,
    out,
    heads: tl.constexpr,
    kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> None:
    pid = tl.program_id(0)
    total = heads * head_dim
    group = heads // kv_heads
    offs = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < total
    head = offs // head_dim
    dim = offs - head * head_dim
    kv_head = head // group
    tl.store(out + offs, tl.load(v + kv_head * head_dim + dim, mask=mask, other=0.0), mask=mask)

@triton.jit
def _matvec_looped_kernel(
    weight,
    x,
    y,
    rows: tl.constexpr,
    cols: tl.constexpr,
    stride_wm: tl.constexpr,
    stride_wn: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> None:
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < rows

    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for start_n in range(0, cols, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < cols
        w = tl.load(
            weight + offs_m[:, None] * stride_wm + offs_n[None, :] * stride_wn,
            mask=mask_m[:, None] & mask_n[None, :],
            other=0.0,
        ).to(tl.float32)
        xv = tl.load(x + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        acc += tl.sum(w * xv[None, :], axis=1)
    tl.store(y + offs_m, acc, mask=mask_m)


@triton.jit
def _scaled_matvec_looped_kernel(
    weight,
    scales,
    x,
    y,
    rows: tl.constexpr,
    cols: tl.constexpr,
    stride_wm: tl.constexpr,
    stride_wn: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> None:
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < rows

    s = tl.load(scales + offs_m, mask=mask_m, other=1.0).to(tl.float32)

    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for start_n in range(0, cols, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < cols
        w = tl.load(
            weight + offs_m[:, None] * stride_wm + offs_n[None, :] * stride_wn,
            mask=mask_m[:, None] & mask_n[None, :],
            other=0.0,
        ).to(tl.float32)
        xv = tl.load(x + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        acc += tl.sum(w * xv[None, :], axis=1)
    tl.store(y + offs_m, acc * s, mask=mask_m)


@triton.jit
def _matvec_split_k_kernel(
    weight,
    x,
    partials,
    rows: tl.constexpr,
    cols: tl.constexpr,
    stride_wm: tl.constexpr,
    stride_wn: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> None:
    row_pid = tl.program_id(0)
    split_pid = tl.program_id(1)
    offs_m = row_pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < rows
    chunk_n: tl.constexpr = (cols + SPLIT_K - 1) // SPLIT_K
    split_start = split_pid * chunk_n
    split_end = split_start + chunk_n
    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for local_n in range(0, chunk_n, BLOCK_N):
        offs_n = split_start + local_n + tl.arange(0, BLOCK_N)
        mask_n = (offs_n < cols) & (offs_n < split_end)
        weight_values = tl.load(
            weight
            + offs_m[:, None] * stride_wm
            + offs_n[None, :] * stride_wn,
            mask=mask_m[:, None] & mask_n[None, :],
            other=0.0,
        ).to(tl.float32)
        x_values = tl.load(
            x + offs_n,
            mask=mask_n,
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(weight_values * x_values[None, :], axis=1)
    tl.store(
        partials + split_pid * rows + offs_m,
        acc,
        mask=mask_m,
    )


@triton.jit
def _reduce_split_k_kernel(
    partials,
    y,
    rows: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
) -> None:
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < rows
    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for split in range(SPLIT_K):
        acc += tl.load(
            partials + split * rows + offs_m,
            mask=mask_m,
            other=0.0,
        )
    tl.store(y + offs_m, acc, mask=mask_m)


@triton.jit
def _reduce_scaled_split_k_kernel(
    partials,
    scales,
    y,
    rows: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
) -> None:
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < rows
    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for split in range(SPLIT_K):
        acc += tl.load(
            partials + split * rows + offs_m,
            mask=mask_m,
            other=0.0,
        )
    scales_values = tl.load(
        scales + offs_m,
        mask=mask_m,
        other=1.0,
    ).to(tl.float32)
    tl.store(y + offs_m, acc * scales_values, mask=mask_m)


@triton.jit
def _block_scaled_matvec_looped_kernel(
    weight,
    scales,
    x,
    y,
    rows: tl.constexpr,
    cols: tl.constexpr,
    stride_wm: tl.constexpr,
    stride_wn: tl.constexpr,
    stride_sm: tl.constexpr,
    stride_sn: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> None:
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < rows
    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for start_n in range(0, cols, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < cols
        weight_block = tl.load(
            weight
            + offs_m[:, None] * stride_wm
            + offs_n[None, :] * stride_wn,
            mask=mask_m[:, None] & mask_n[None, :],
            other=0.0,
        ).to(tl.float32)
        vector_block = tl.load(
            x + offs_n, mask=mask_n, other=0.0
        ).to(tl.float32)
        scale = tl.load(
            scales
            + offs_m * stride_sm
            + (start_n // BLOCK_N) * stride_sn,
            mask=mask_m,
            other=1.0,
        ).to(tl.float32)
        acc += tl.sum(weight_block * vector_block[None, :], axis=1) * scale
    tl.store(y + offs_m, acc, mask=mask_m)


@triton.jit
def _fused_gate_up_silu_kernel(
    gate_weight,
    up_weight,
    x,
    out,
    rows: tl.constexpr,
    cols: tl.constexpr,
    stride_gate_m: tl.constexpr,
    stride_gate_n: tl.constexpr,
    stride_up_m: tl.constexpr,
    stride_up_n: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> None:
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    mask_m = offs_m < rows
    mask_n = offs_n < cols

    w_gate = tl.load(
        gate_weight + offs_m[:, None] * stride_gate_m + offs_n[None, :] * stride_gate_n,
        mask=mask_m[:, None] & mask_n[None, :],
        other=0.0,
    ).to(tl.float32)

    w_up = tl.load(
        up_weight + offs_m[:, None] * stride_up_m + offs_n[None, :] * stride_up_n,
        mask=mask_m[:, None] & mask_n[None, :],
        other=0.0,
    ).to(tl.float32)

    xv = tl.load(x + offs_n, mask=mask_n, other=0.0).to(tl.float32)

    gate_val = tl.sum(w_gate * xv[None, :], axis=1).to(tl.bfloat16).to(tl.float32)
    up_val = tl.sum(w_up * xv[None, :], axis=1).to(tl.bfloat16).to(tl.float32)

    silu_val = gate_val / (1.0 + tl.exp(-gate_val))
    out_val = silu_val * up_val

    tl.store(out + offs_m, out_val, mask=mask_m)


@triton.jit
def _fused_gate_up_silu_nomask_n_kernel(
    gate_weight,
    up_weight,
    x,
    out,
    rows: tl.constexpr,
    cols: tl.constexpr,
    stride_gate_m: tl.constexpr,
    stride_gate_n: tl.constexpr,
    stride_up_m: tl.constexpr,
    stride_up_n: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> None:
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    mask_m = offs_m < rows

    w_gate = tl.load(
        gate_weight + offs_m[:, None] * stride_gate_m + offs_n[None, :] * stride_gate_n,
        mask=mask_m[:, None],
        other=0.0,
    ).to(tl.float32)

    w_up = tl.load(
        up_weight + offs_m[:, None] * stride_up_m + offs_n[None, :] * stride_up_n,
        mask=mask_m[:, None],
        other=0.0,
    ).to(tl.float32)

    xv = tl.load(x + offs_n).to(tl.float32)

    gate_val = tl.sum(w_gate * xv[None, :], axis=1).to(tl.bfloat16).to(tl.float32)
    up_val = tl.sum(w_up * xv[None, :], axis=1).to(tl.bfloat16).to(tl.float32)

    silu_val = gate_val / (1.0 + tl.exp(-gate_val))
    out_val = silu_val * up_val

    tl.store(out + offs_m, out_val, mask=mask_m)


@triton.jit
def _fused_scaled_gate_up_silu_kernel(
    gate_weight,
    gate_scales,
    up_weight,
    up_scales,
    x,
    out,
    rows: tl.constexpr,
    cols: tl.constexpr,
    stride_gate_m: tl.constexpr,
    stride_gate_n: tl.constexpr,
    stride_up_m: tl.constexpr,
    stride_up_n: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> None:
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    mask_m = offs_m < rows
    mask_n = offs_n < cols
    matrix_mask = mask_m[:, None] & mask_n[None, :]

    gate_values = tl.load(
        gate_weight
        + offs_m[:, None] * stride_gate_m
        + offs_n[None, :] * stride_gate_n,
        mask=matrix_mask,
        other=0.0,
    ).to(tl.float32)
    up_values = tl.load(
        up_weight
        + offs_m[:, None] * stride_up_m
        + offs_n[None, :] * stride_up_n,
        mask=matrix_mask,
        other=0.0,
    ).to(tl.float32)
    x_values = tl.load(
        x + offs_n,
        mask=mask_n,
        other=0.0,
    ).to(tl.float32)
    gate_scale = tl.load(
        gate_scales + offs_m,
        mask=mask_m,
        other=1.0,
    ).to(tl.float32)
    up_scale = tl.load(
        up_scales + offs_m,
        mask=mask_m,
        other=1.0,
    ).to(tl.float32)
    # Match the retained three-kernel path, which stores both projections to
    # BF16 before the SiLU/multiply kernel reloads them as FP32.
    gate = (
        tl.sum(gate_values * x_values[None, :], axis=1) * gate_scale
    ).to(tl.bfloat16).to(tl.float32)
    up = (
        tl.sum(up_values * x_values[None, :], axis=1) * up_scale
    ).to(tl.bfloat16).to(tl.float32)
    silu = gate / (1.0 + tl.exp(-gate))
    tl.store(out + offs_m, silu * up, mask=mask_m)


@triton.jit
def _fused_scaled_gate_up_silu_nomask_n_kernel(
    gate_weight,
    gate_scales,
    up_weight,
    up_scales,
    x,
    out,
    rows: tl.constexpr,
    cols: tl.constexpr,
    stride_gate_m: tl.constexpr,
    stride_gate_n: tl.constexpr,
    stride_up_m: tl.constexpr,
    stride_up_n: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> None:
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    mask_m = offs_m < rows

    gate_values = tl.load(
        gate_weight
        + offs_m[:, None] * stride_gate_m
        + offs_n[None, :] * stride_gate_n,
        mask=mask_m[:, None],
        other=0.0,
    ).to(tl.float32)
    up_values = tl.load(
        up_weight
        + offs_m[:, None] * stride_up_m
        + offs_n[None, :] * stride_up_n,
        mask=mask_m[:, None],
        other=0.0,
    ).to(tl.float32)
    x_values = tl.load(x + offs_n).to(tl.float32)
    gate_scale = tl.load(
        gate_scales + offs_m,
        mask=mask_m,
        other=1.0,
    ).to(tl.float32)
    up_scale = tl.load(
        up_scales + offs_m,
        mask=mask_m,
        other=1.0,
    ).to(tl.float32)
    gate = (
        tl.sum(gate_values * x_values[None, :], axis=1) * gate_scale
    ).to(tl.bfloat16).to(tl.float32)
    up = (
        tl.sum(up_values * x_values[None, :], axis=1) * up_scale
    ).to(tl.bfloat16).to(tl.float32)
    silu = gate / (1.0 + tl.exp(-gate))
    tl.store(out + offs_m, silu * up, mask=mask_m)


@dataclass
class TritonDecodeBuffers:
    hidden_a: torch.Tensor
    hidden_b: torch.Tensor
    normed: torch.Tensor
    qkv: torch.Tensor
    q_norm: torch.Tensor
    k_norm: torch.Tensor
    attn: torch.Tensor
    attn_out: torch.Tensor
    gate_up: torch.Tensor
    mlp_act: torch.Tensor
    mlp: torch.Tensor
    final: torch.Tensor


def select_matvec_config(rows: int, cols: int) -> tuple[int, int]:
    """Small static launch table established by scripts/profile_raw_forward_shapes.py."""
    if rows >= 65536:
        return 64, 4
    if cols >= 3072:
        return 64, 4
    if cols <= 1024 and rows >= cols * 3:
        return 8, 8
    if rows <= 1024 and cols >= 2048:
        return 8, 4
    if rows == cols * 2 and cols <= 1024:
        return 32, 8
    return 16, 8


def _select_block_m(rows: int, cols: int) -> int:
    return select_matvec_config(rows, cols)[0]


class TritonDecodeBackend:
    def __init__(
        self,
        device: torch.device,
        dtype: torch.dtype,
        hidden_size: int,
        q_dim: int,
        kv_dim: int,
        intermediate_size: int,
        argmax_mode: str = "torch",  # torch | triton_two_stage
    ) -> None:
        self.device = device
        self.dtype = dtype
        self.hidden_size = hidden_size
        self.q_dim = q_dim
        self.kv_dim = kv_dim
        self.intermediate_size = intermediate_size
        self.argmax_mode = argmax_mode

        self.buffers = TritonDecodeBuffers(
            hidden_a=torch.empty(hidden_size, device=device, dtype=dtype),
            hidden_b=torch.empty(hidden_size, device=device, dtype=dtype),
            normed=torch.empty(hidden_size, device=device, dtype=dtype),
            qkv=torch.empty(q_dim + 2 * kv_dim, device=device, dtype=dtype),
            q_norm=torch.empty(q_dim, device=device, dtype=dtype),
            k_norm=torch.empty(kv_dim, device=device, dtype=dtype),
            attn=torch.empty(q_dim, device=device, dtype=dtype),
            attn_out=torch.empty(hidden_size, device=device, dtype=dtype),
            gate_up=torch.empty(2 * intermediate_size, device=device, dtype=dtype),
            mlp_act=torch.empty(intermediate_size, device=device, dtype=dtype),
            mlp=torch.empty(hidden_size, device=device, dtype=dtype),
            final=torch.empty(hidden_size, device=device, dtype=dtype),
        )

        # For full-logits greedy argmax. This is tiny for Qwen3 0.6B:
        # 151936 logits * 2 bytes ~= 304KB.
        self._logits_buffer: torch.Tensor | None = None

        # For custom Triton two-stage argmax.
        # int32 is enough for vocab indices and much cheaper than int64 partials.
        self._argmax_partial_vals: torch.Tensor | None = None
        self._argmax_partial_idxs: torch.Tensor | None = None
        self._argmax_out_idx = torch.empty((), device=device, dtype=torch.int64)
        self._argmax_out_val = torch.empty((), device=device, dtype=torch.float32)
        # For custom Triton persistent argmax.
        self._persistent_argmax_counter = torch.zeros((), device=device, dtype=torch.int32)
        self._persistent_argmax_partial_vals = None
        self._persistent_argmax_partial_idxs = None
        self._split_k_partials: torch.Tensor | None = None
        self._attention_partial_max: torch.Tensor | None = None
        self._attention_partial_den: torch.Tensor | None = None
        self._attention_partial_acc: torch.Tensor | None = None
        self._matvec_launch_plans: dict[Any, tuple[object, ...]] = {}
        self._multi_launch_plans: dict[tuple[int, ...], tuple[int, ...]] = {}
        self._repeat_matvec_plans: dict[int, tuple[int, ...]] = {}
        self._fused_gate_up_silu_plans: dict[Any, tuple[object, ...]] = {}
        self._fused_scaled_gate_up_silu_plans: dict[
            Any,
            tuple[object, ...],
        ] = {}

    def matvec(
        self,
        weight: torch.Tensor,
        x: torch.Tensor,
        out: torch.Tensor,
        *,
        block_m: int | None = None,
        num_warps: int | None = None,
        config_name: str | None = None,
    ) -> torch.Tensor:
        cache_key = (id(weight), config_name, block_m, num_warps)
        plan = self._matvec_launch_plans.get(cache_key)
        if plan is None or block_m is not None or num_warps is not None:
            rows = int(weight.shape[0])
            cols = int(weight.shape[1])
            selected_block_m, selected_warps = select_matvec_config(rows, cols)
            if weight.is_floating_point() and weight.element_size() == 1 and rows >= 65536:
                selected_block_m, selected_warps = 32, 8
            block_m = selected_block_m if block_m is None else block_m
            num_warps = selected_warps if num_warps is None else num_warps
            if config_name is not None and config_name.startswith("row_block_m"):
                block_m = int(config_name.removeprefix("row_block_m"))
            
            is_loop = False
            loop_block_n = None
            if config_name is not None and config_name.startswith("triton_loop_"):
                is_loop = True
                loop_block_n = int(config_name.split("_")[-1])
            
            if is_loop:
                kernel = _matvec_looped_kernel
                block_n = loop_block_n
            else:
                block_n = triton.next_power_of_2(cols)
                kernel = _matvec_nomask_n_kernel if cols == block_n else _matvec_kernel

            plan = (
                kernel,
                rows,
                cols,
                block_n,
                block_m,
                num_warps,
                int(weight.stride(0)),
                int(weight.stride(1)),
            )
            if block_m == selected_block_m and num_warps == selected_warps and config_name is None:
                self._matvec_launch_plans[cache_key] = plan
        kernel, rows, cols, block_n, block_m, num_warps, stride0, stride1 = plan

        kernel[(triton.cdiv(rows, block_m),)](
            weight,
            x,
            out,
            rows,
            cols,
            stride0,
            stride1,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            num_warps=num_warps,
        )
        return out

    def split_k_matvec(
        self,
        weight: torch.Tensor,
        x: torch.Tensor,
        out: torch.Tensor,
        *,
        scales: torch.Tensor | None = None,
        split_k: int = 4,
        block_m: int = 32,
        block_n: int = 256,
        num_warps: int = 4,
    ) -> torch.Tensor:
        if split_k not in {2, 4, 8}:
            raise ValueError("split_k must be 2, 4, or 8")
        rows = int(weight.shape[0])
        cols = int(weight.shape[1])
        if scales is not None and (
            scales.ndim != 1 or scales.numel() != rows
        ):
            raise ValueError(
                "split-K scaled matvec requires one scale per output row"
            )
        partial_count = split_k * rows
        if (
            self._split_k_partials is None
            or self._split_k_partials.numel() < partial_count
        ):
            self._split_k_partials = torch.empty(
                partial_count,
                device=self.device,
                dtype=torch.float32,
            )
        partials = self._split_k_partials[:partial_count]
        _matvec_split_k_kernel[
            (triton.cdiv(rows, block_m), split_k)
        ](
            weight,
            x,
            partials,
            rows,
            cols,
            int(weight.stride(0)),
            int(weight.stride(1)),
            SPLIT_K=split_k,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            num_warps=num_warps,
        )
        reduce_kernel = (
            _reduce_scaled_split_k_kernel
            if scales is not None
            else _reduce_split_k_kernel
        )
        if scales is None:
            reduce_kernel[(triton.cdiv(rows, block_m),)](
                partials,
                out,
                rows,
                SPLIT_K=split_k,
                BLOCK_M=block_m,
                num_warps=4,
            )
        else:
            reduce_kernel[(triton.cdiv(rows, block_m),)](
                partials,
                scales,
                out,
                rows,
                SPLIT_K=split_k,
                BLOCK_M=block_m,
                num_warps=4,
            )
        return out

    def mxfp4_selected_matvec(
        self,
        blocks: torch.Tensor,
        scales: torch.Tensor,
        x: torch.Tensor,
        expert_indices: torch.Tensor,
        out: torch.Tensor,
        *,
        block_m: int = 4,
        num_warps: int = 8,
    ) -> torch.Tensor:
        if blocks.ndim != 4 or scales.ndim != 3:
            raise ValueError(
                "MXFP4 expert blocks/scales must be rank 4/rank 3"
            )
        if blocks.shape[:-1] != scales.shape:
            raise ValueError(
                f"MXFP4 blocks/scales mismatch: {blocks.shape} vs {scales.shape}"
            )
        rows = int(blocks.shape[1])
        cols = int(blocks.shape[2] * blocks.shape[3] * 2)
        selected = int(expert_indices.numel())
        if x.ndim == 1:
            if int(x.numel()) != cols:
                raise ValueError(
                    f"MXFP4 vector has {x.numel()} values, expected {cols}"
                )
            stride_xe = 0
        elif x.ndim == 2 and tuple(x.shape) == (selected, cols):
            stride_xe = int(x.stride(0))
        else:
            raise ValueError(
                f"MXFP4 input must be {(cols,)} or {(selected, cols)}, "
                f"got {tuple(x.shape)}"
            )
        if tuple(out.shape) != (selected, rows):
            raise ValueError(
                f"MXFP4 output must be {(selected, rows)}, got {tuple(out.shape)}"
            )
        block_n = triton.next_power_of_2(cols)
        _mxfp4_selected_matvec_kernel[
            (triton.cdiv(rows, block_m), selected)
        ](
            blocks,
            scales,
            x,
            expert_indices,
            out,
            rows=rows,
            cols=cols,
            stride_be=blocks.stride(0),
            stride_br=blocks.stride(1),
            stride_bg=blocks.stride(2),
            stride_bb=blocks.stride(3),
            stride_se=scales.stride(0),
            stride_sr=scales.stride(1),
            stride_sg=scales.stride(2),
            stride_xe=stride_xe,
            stride_oe=out.stride(0),
            stride_or=out.stride(1),
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            num_warps=num_warps,
        )
        return out

    def fused_gate_up_silu(
        self,
        gate_weight: torch.Tensor,
        up_weight: torch.Tensor,
        x: torch.Tensor,
        out: torch.Tensor,
        *,
        block_m: int | None = None,
        num_warps: int | None = None,
    ) -> torch.Tensor:
        cache_key = (id(gate_weight), id(up_weight), block_m, num_warps)
        plan = self._fused_gate_up_silu_plans.get(cache_key)
        if plan is None or block_m is not None or num_warps is not None:
            rows = int(gate_weight.shape[0])
            cols = int(gate_weight.shape[1])
            block_n = triton.next_power_of_2(cols)
            selected_block_m, selected_warps = select_matvec_config(rows, cols)
            block_m = selected_block_m if block_m is None else block_m
            num_warps = selected_warps if num_warps is None else num_warps
            kernel = _fused_gate_up_silu_nomask_n_kernel if cols == block_n else _fused_gate_up_silu_kernel
            plan = (
                kernel,
                rows,
                cols,
                block_n,
                block_m,
                num_warps,
                int(gate_weight.stride(0)),
                int(gate_weight.stride(1)),
                int(up_weight.stride(0)),
                int(up_weight.stride(1)),
            )
            if block_m == selected_block_m and num_warps == selected_warps:
                self._fused_gate_up_silu_plans[cache_key] = plan

        kernel, rows, cols, block_n, block_m, num_warps, stride_gate_m, stride_gate_n, stride_up_m, stride_up_n = plan

        kernel[(triton.cdiv(rows, block_m),)](
            gate_weight,
            up_weight,
            x,
            out,
            rows,
            cols,
            stride_gate_m,
            stride_gate_n,
            stride_up_m,
            stride_up_n,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            num_warps=num_warps,
        )
        return out

    def fused_scaled_gate_up_silu(
        self,
        gate_weight: torch.Tensor,
        gate_scales: torch.Tensor,
        up_weight: torch.Tensor,
        up_scales: torch.Tensor,
        x: torch.Tensor,
        out: torch.Tensor,
        *,
        block_m: int | None = None,
        num_warps: int | None = None,
    ) -> torch.Tensor:
        if gate_scales.ndim != 1 or up_scales.ndim != 1:
            raise ValueError(
                "fused scaled gate/up requires one scale per output row"
            )
        cache_key = (
            id(gate_weight),
            id(gate_scales),
            id(up_weight),
            id(up_scales),
            block_m,
            num_warps,
        )
        plan = self._fused_scaled_gate_up_silu_plans.get(cache_key)
        if plan is None or block_m is not None or num_warps is not None:
            rows = int(gate_weight.shape[0])
            cols = int(gate_weight.shape[1])
            if tuple(up_weight.shape) != tuple(gate_weight.shape):
                raise ValueError(
                    "fused scaled gate/up matrices must have equal shapes"
                )
            if gate_scales.numel() != rows or up_scales.numel() != rows:
                raise ValueError(
                    "fused scaled gate/up scales must match matrix rows"
                )
            block_n = triton.next_power_of_2(cols)
            selected_block_m, selected_warps = select_matvec_config(
                rows,
                cols,
            )
            block_m = selected_block_m if block_m is None else block_m
            num_warps = selected_warps if num_warps is None else num_warps
            kernel = (
                _fused_scaled_gate_up_silu_nomask_n_kernel
                if cols == block_n
                else _fused_scaled_gate_up_silu_kernel
            )
            plan = (
                kernel,
                rows,
                cols,
                block_n,
                block_m,
                num_warps,
                int(gate_weight.stride(0)),
                int(gate_weight.stride(1)),
                int(up_weight.stride(0)),
                int(up_weight.stride(1)),
            )
            if (
                block_m == selected_block_m
                and num_warps == selected_warps
            ):
                self._fused_scaled_gate_up_silu_plans[cache_key] = plan

        (
            kernel,
            rows,
            cols,
            block_n,
            block_m,
            num_warps,
            stride_gate_m,
            stride_gate_n,
            stride_up_m,
            stride_up_n,
        ) = plan
        kernel[(triton.cdiv(rows, block_m),)](
            gate_weight,
            gate_scales,
            up_weight,
            up_scales,
            x,
            out,
            rows,
            cols,
            stride_gate_m,
            stride_gate_n,
            stride_up_m,
            stride_up_n,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            num_warps=num_warps,
        )
        return out

    def logits_buffer(self, rows: int, dtype: torch.dtype) -> torch.Tensor:
        if (
            self._logits_buffer is None
            or self._logits_buffer.numel() != rows
            or self._logits_buffer.dtype != dtype
        ):
            self._logits_buffer = torch.empty(rows, device=self.device, dtype=dtype)
        return self._logits_buffer

    def scaled_matvec(
        self,
        weight: torch.Tensor,
        scales: torch.Tensor,
        x: torch.Tensor,
        out: torch.Tensor,
        config_name: str | None = None,
        *,
        block_m: int | None = None,
        num_warps: int | None = None,
    ) -> torch.Tensor:
        rows = int(weight.shape[0])
        cols = int(weight.shape[1])
        if scales.dim() == 2:
            scale_blocks = int(scales.shape[1])
            scale_block_n = triton.next_power_of_2(
                (cols + scale_blocks - 1) // scale_blocks
            )
            selected_block_m, selected_warps = select_matvec_config(
                rows,
                cols,
            )
            block_m = (
                selected_block_m if block_m is None else block_m
            )
            num_warps = (
                selected_warps if num_warps is None else num_warps
            )
            _block_scaled_matvec_looped_kernel[
                (triton.cdiv(rows, block_m),)
            ](
                weight,
                scales,
                x,
                out,
                rows,
                cols,
                int(weight.stride(0)),
                int(weight.stride(1)),
                int(scales.stride(0)),
                int(scales.stride(1)),
                BLOCK_M=block_m,
                BLOCK_N=scale_block_n,
                num_warps=num_warps,
            )
            return out
        tensorcore_all = (
            os.environ.get("THINTENSOR_TENSORCORE_MATVEC", "0") == "1"
        )
        tensorcore_int8_down = (
            os.environ.get("THINTENSOR_INT8_DOWN_TENSORCORE", "0") == "1"
            and weight.dtype == torch.int8
            and rows == 2048
            and cols == 11008
        )
        tensorcore_int8 = (
            os.environ.get("THINTENSOR_INT8_TENSORCORE", "0") == "1"
            and weight.dtype == torch.int8
        )
        if (
            (tensorcore_all or tensorcore_int8 or tensorcore_int8_down)
            and
            x.dtype == torch.bfloat16
            and cols % 256 == 0
            and weight.dtype in {torch.float8_e4m3fn, torch.int8}
        ):
            tensorcore_kernel = (
                _fp8_bf16_tensorcore_matvec_kernel
                if weight.dtype == torch.float8_e4m3fn
                else _int8_bf16_tensorcore_matvec_kernel
            )
            tensorcore_block_n = (
                8
                if weight.dtype == torch.float8_e4m3fn
                else int(os.environ.get("THINTENSOR_INT8_TC_BLOCK_N", "1"))
            )
            tensorcore_block_m = int(
                os.environ.get(
                    "THINTENSOR_INT8_TC_BLOCK_M",
                    "32" if rows <= 512 else "64",
                )
            )
            tensorcore_block_k = int(
                os.environ.get("THINTENSOR_INT8_TC_BLOCK_K", "256")
            )
            if cols % tensorcore_block_k:
                tensorcore_block_k = 256
            tensorcore_kernel[
                (triton.cdiv(rows, tensorcore_block_m),)
            ](
                weight,
                scales,
                x,
                out,
                rows,
                cols,
                int(weight.stride(0)),
                BLOCK_M=tensorcore_block_m,
                BLOCK_K=tensorcore_block_k,
                BLOCK_N=tensorcore_block_n,
                num_warps=4,
                num_stages=3,
            )
            return out
        block_n = triton.next_power_of_2(cols)
        selected_block_m, selected_warps = select_matvec_config(rows, cols)
        block_m = selected_block_m if block_m is None else block_m
        num_warps = selected_warps if num_warps is None else num_warps
        if weight.element_size() == 1 and rows >= 65536:
            block_m, num_warps = 32, 8
            
        is_loop = False
        loop_block_n = None
        if config_name is not None and config_name.startswith("triton_loop_"):
            is_loop = True
            loop_block_n = int(config_name.split("_")[-1])

        if is_loop:
            kernel = _scaled_matvec_looped_kernel
            block_n = loop_block_n
        else:
            kernel = (
                _scaled_matvec_nomask_n_kernel
                if cols == block_n
                else _scaled_matvec_kernel
            )

        kernel[(triton.cdiv(rows, block_m),)](
            weight,
            scales,
            x,
            out,
            rows,
            cols,
            int(weight.stride(0)),
            int(weight.stride(1)),
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            num_warps=num_warps,
        )
        return out

    def int4_scaled_matvec(
        self,
        packed_weight: torch.Tensor,
        scales: torch.Tensor,
        x: torch.Tensor,
        out: torch.Tensor,
        *,
        rows: int,
        cols: int,
        group_size: int,
    ) -> torch.Tensor:
        use_dot = (
            os.environ.get("THINTENSOR_INT4_DOT", "0") == "1"
            and group_size >= cols
        )
        if use_dot:
            kernel = _int4_dot_matvec_kernel
            block_n = 256
            block_m = 16
            num_warps = 4
        elif cols > 4096:
            kernel = _int4_scaled_matvec_looped_kernel
            block_n = int(
                os.environ.get("THINTENSOR_INT4_DOWN_BLOCK_N", "256")
            )
            block_m = int(
                os.environ.get("THINTENSOR_INT4_DOWN_BLOCK_M", "16")
            )
            num_warps = int(
                os.environ.get("THINTENSOR_INT4_DOWN_WARPS", "4")
            )
        else:
            kernel = _int4_scaled_matvec_kernel
            block_n = triton.next_power_of_2((cols + 1) // 2)
            selected_block_m, selected_warps = select_matvec_config(
                rows, cols
            )
            role = "HEAD" if rows >= 65536 else "DIRECT"
            block_m = int(
                os.environ.get(
                    f"THINTENSOR_INT4_{role}_BLOCK_M",
                    str(selected_block_m),
                )
            )
            num_warps = int(
                os.environ.get(
                    f"THINTENSOR_INT4_{role}_WARPS",
                    str(selected_warps),
                )
            )
        if use_dot:
            kernel[(triton.cdiv(rows, block_m),)](
                packed_weight,
                scales,
                x,
                out,
                rows,
                cols,
                int(packed_weight.stride(0)),
                int(scales.stride(0)),
                BLOCK_M=block_m,
                BLOCK_K=block_n,
                BLOCK_OUT=16,
                num_warps=num_warps,
            )
        else:
            kernel[(triton.cdiv(rows, block_m),)](
                packed_weight,
                scales,
                x,
                out,
                rows,
                cols,
                int(packed_weight.stride(0)),
                int(scales.stride(0)),
                int(scales.stride(1)),
                GROUP_SIZE=group_size,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                num_warps=num_warps,
            )
        return out

    def mxfp4_tensorcore_matvec(
        self,
        packed_weight: torch.Tensor,
        scales: torch.Tensor,
        x: torch.Tensor,
        out: torch.Tensor,
        *,
        rows: int,
        cols: int,
        residual_bits: torch.Tensor | None = None,
        residual_scales: torch.Tensor | None = None,
        post_scales: torch.Tensor | None = None,
        block_m: int = 64,
        block_k: int = 256,
        block_n: int = 8,
    ) -> torch.Tensor:
        if packed_weight.dtype != torch.uint8:
            raise ValueError("packed FP4 weights must be uint8")
        if scales.dtype != torch.uint8:
            raise ValueError("MXFP4 E8M0 scales must be uint8")
        if tuple(packed_weight.shape) != (rows, cols // 2):
            raise ValueError(
                "packed MXFP4 weight shape must be "
                f"{(rows, cols // 2)}, got {tuple(packed_weight.shape)}"
            )
        if tuple(scales.shape) != (rows, cols // 32):
            raise ValueError(
                "MXFP4 scale shape must be "
                f"{(rows, cols // 32)}, got {tuple(scales.shape)}"
            )
        if cols % block_k or block_k % 32:
            raise ValueError("MXFP4 tensor-core K dimensions must be 32-aligned")
        has_binary_residual = residual_bits is not None
        if has_binary_residual != (residual_scales is not None):
            raise ValueError("binary residual bits and scales must be provided together")
        if has_binary_residual:
            assert residual_bits is not None and residual_scales is not None
            if tuple(residual_bits.shape) != (rows, cols // 8):
                raise ValueError("binary residual bits shape mismatch")
            if tuple(residual_scales.shape) != (rows, cols // 32):
                raise ValueError("binary residual scales shape mismatch")
            if residual_scales.dtype != torch.uint8:
                raise ValueError("binary residual scales must use E8M0 uint8")
        if post_scales is not None and tuple(post_scales.shape) != (rows,):
            raise ValueError("MXFP4 post-scale shape mismatch")
        _mxfp4_bf16_tensorcore_matvec_kernel[
            (triton.cdiv(rows, block_m),)
        ](
            packed_weight,
            scales,
            residual_bits if residual_bits is not None else packed_weight,
            residual_scales if residual_scales is not None else scales,
            post_scales if post_scales is not None else scales,
            x,
            out,
            rows,
            cols,
            int(packed_weight.stride(0)),
            int(scales.stride(0)),
            int(scales.stride(1)),
            BLOCK_M=block_m,
            BLOCK_K=block_k,
            BLOCK_N=block_n,
            HAS_BINARY_RESIDUAL=has_binary_residual,
            HAS_POST_SCALE=post_scales is not None,
            num_warps=4,
            num_stages=3,
        )
        return out

    def block_hadamard_32(
        self,
        x: torch.Tensor,
        out: torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim != 1 or out.shape != x.shape:
            raise ValueError("block Hadamard input/output must be equal 1D tensors")
        cols = int(x.numel())
        if cols % 32:
            raise ValueError("block Hadamard requires a multiple of 32 values")
        _block_hadamard_32_kernel[(triton.cdiv(cols, 32),)](
            x,
            out,
            cols=cols,
            num_warps=1,
        )
        return out

    def block_hadamard(
        self,
        x: torch.Tensor,
        out: torch.Tensor,
        *,
        block_size: int,
        signs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.ndim != 1 or out.shape != x.shape:
            raise ValueError("block Hadamard input/output must be equal 1D tensors")
        cols = int(x.numel())
        if block_size < 2 or block_size > 2048 or block_size & (block_size - 1):
            raise ValueError("Hadamard block size must be a power of two in [2, 2048]")
        if cols % block_size:
            raise ValueError(
                f"Hadamard block size {block_size} does not divide {cols}"
            )
        if signs is not None and signs.shape != x.shape:
            raise ValueError("Hadamard signs must match the input shape")
        _block_hadamard_kernel[(cols // block_size,)](
            x,
            signs if signs is not None else x,
            out,
            cols=cols,
            BLOCK_SIZE=block_size,
            SIGNED=signs is not None,
            num_warps=8 if block_size >= 512 else 4,
        )
        return out

    def multi_matvec(
        self,
        weights: tuple[torch.Tensor, ...],
        x: torch.Tensor,
        out: torch.Tensor,
        *,
        block_m: int = 8,
        num_warps: int = 8,
    ) -> torch.Tensor:
        if len(weights) not in {2, 3}:
            raise ValueError("multi_matvec supports exactly two or three matrices")
        weight0, weight1 = weights[:2]
        weight2 = weights[2] if len(weights) == 3 else weight1
        cache_key = tuple(id(weight) for weight in weights) + (block_m, num_warps)
        plan = self._multi_launch_plans.get(cache_key)
        if plan is None:
            cols = int(weight0.shape[1])
            if triton.next_power_of_2(cols) != cols:
                # Correct fallback for any non-power-of-two hidden size.
                # Keep correctness by dispatching normal matvecs into slices of the fused output.
                offset = 0
                for weight in weights:
                    rows_i = int(weight.shape[0])
                    self.matvec(weight, x, out[offset : offset + rows_i])
                    offset += rows_i
                return out
            if any(int(weight.shape[1]) != cols for weight in weights):
                raise ValueError("multi_matvec matrices must have the same column count")
            rows0 = int(weight0.shape[0])
            rows1 = int(weight1.shape[0])
            rows2 = int(weight2.shape[0]) if len(weights) == 3 else 0
            blocks0 = triton.cdiv(rows0, block_m)
            blocks1 = triton.cdiv(rows1, block_m)
            blocks2 = triton.cdiv(rows2, block_m)
            plan = (
                cols,
                rows0,
                rows1,
                rows2,
                blocks0,
                blocks1,
                blocks2,
                int(weight0.stride(0)),
                int(weight1.stride(0)),
                int(weight2.stride(0)),
            )
            self._multi_launch_plans[cache_key] = plan
        cols, rows0, rows1, rows2, blocks0, blocks1, blocks2, stride0, stride1, stride2 = plan
        _multi_matvec_nomask_n_kernel[(blocks0 + blocks1 + blocks2,)](
            weight0,
            weight1,
            weight2,
            x,
            out,
            rows0,
            rows1,
            rows2,
            cols,
            stride0,
            stride1,
            stride2,
            BLOCKS0=blocks0,
            BLOCKS1=blocks1,
            BLOCK_M=block_m,
            BLOCK_N=cols,
            num_warps=num_warps,
        )
        return out

    def multi_int4_scaled_matvec(
        self,
        weights: tuple[torch.Tensor, ...],
        scales: tuple[torch.Tensor, ...],
        rows: tuple[int, ...],
        cols: int,
        x: torch.Tensor,
        out: torch.Tensor,
    ) -> torch.Tensor:
        if len(weights) not in {2, 3} or len(scales) != len(weights):
            raise ValueError(
                "multi INT4 matvec supports two or three matrices"
            )
        block_m = 16
        num_warps = 8
        weight0, weight1 = weights[:2]
        weight2 = weights[2] if len(weights) == 3 else weight1
        scale0, scale1 = scales[:2]
        scale2 = scales[2] if len(scales) == 3 else scale1
        rows0, rows1 = rows[:2]
        rows2 = rows[2] if len(rows) == 3 else 0
        blocks0 = triton.cdiv(rows0, block_m)
        blocks1 = triton.cdiv(rows1, block_m)
        blocks2 = triton.cdiv(rows2, block_m)
        _multi_int4_scaled_matvec_kernel[
            (blocks0 + blocks1 + blocks2,)
        ](
            weight0,
            weight1,
            weight2,
            scale0,
            scale1,
            scale2,
            x,
            out,
            rows0,
            rows1,
            rows2,
            cols,
            int(weight0.stride(0)),
            int(weight1.stride(0)),
            int(weight2.stride(0)),
            int(scale0.stride(0)),
            int(scale1.stride(0)),
            int(scale2.stride(0)),
            BLOCKS0=blocks0,
            BLOCKS1=blocks1,
            BLOCK_M=block_m,
            BLOCK_B=triton.next_power_of_2((cols + 1) // 2),
            num_warps=num_warps,
        )
        return out

    def multi_scaled_tensorcore_matvec(
        self,
        weights: tuple[torch.Tensor, ...],
        scales: tuple[torch.Tensor, ...],
        rows: tuple[int, ...],
        cols: int,
        x: torch.Tensor,
        out: torch.Tensor,
    ) -> torch.Tensor:
        if len(weights) not in {2, 3} or len(scales) != len(weights):
            raise ValueError(
                "multi scaled tensor-core matvec supports two or three matrices"
            )
        if cols % 256:
            raise ValueError("tensor-core matvec columns must be divisible by 256")
        dtype = weights[0].dtype
        if dtype not in {torch.float8_e4m3fn, torch.int8}:
            raise ValueError("tensor-core matvec requires E4M3 or INT8 weights")
        if any(weight.dtype != dtype for weight in weights):
            raise ValueError("multi tensor-core weights must share one dtype")
        block_m = 64
        block_n = 8 if dtype == torch.float8_e4m3fn else 1
        weight0, weight1 = weights[:2]
        weight2 = weights[2] if len(weights) == 3 else weight1
        scale0, scale1 = scales[:2]
        scale2 = scales[2] if len(scales) == 3 else scale1
        rows0, rows1 = rows[:2]
        rows2 = rows[2] if len(rows) == 3 else 0
        blocks0 = triton.cdiv(rows0, block_m)
        blocks1 = triton.cdiv(rows1, block_m)
        blocks2 = triton.cdiv(rows2, block_m)
        _multi_scaled_tensorcore_matvec_kernel[
            (blocks0 + blocks1 + blocks2,)
        ](
            weight0,
            weight1,
            weight2,
            scale0,
            scale1,
            scale2,
            x,
            out,
            rows0,
            rows1,
            rows2,
            cols,
            int(weight0.stride(0)),
            int(weight1.stride(0)),
            int(weight2.stride(0)),
            int(scale0.stride(0)),
            int(scale1.stride(0)),
            int(scale2.stride(0)),
            BLOCKS0=blocks0,
            BLOCKS1=blocks1,
            BLOCK_M=block_m,
            BLOCK_K=256,
            BLOCK_N=block_n,
            IS_FP8=dtype == torch.float8_e4m3fn,
            num_warps=4,
            num_stages=3,
        )
        return out

    def repeat_kv_matvec(
        self,
        weight: torch.Tensor,
        value: torch.Tensor,
        out: torch.Tensor,
        heads: int,
        kv_heads: int,
        head_dim: int,
    ) -> torch.Tensor:
        plan = self._repeat_matvec_plans.get(id(weight))
        if plan is None:
            rows = int(weight.shape[0])
            cols = int(weight.shape[1])
            block_n = triton.next_power_of_2(cols)
            if block_n != cols:
                raise ValueError("repeat_kv_matvec requires power-of-two columns")
            block_m, num_warps = select_matvec_config(rows, cols)
            plan = (
                rows,
                cols,
                block_n,
                block_m,
                num_warps,
                int(weight.stride(0)),
                int(weight.stride(1)),
            )
            self._repeat_matvec_plans[id(weight)] = plan
        rows, cols, block_n, block_m, num_warps, stride0, stride1 = plan
        _repeat_kv_matvec_nomask_n_kernel[(triton.cdiv(rows, block_m),)](
            weight,
            value,
            out,
            rows,
            cols,
            stride0,
            stride1,
            heads,
            kv_heads,
            head_dim,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            num_warps=num_warps,
        )
        return out

    def single_token_gqa_attention(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        out: torch.Tensor,
        heads: int,
        kv_heads: int,
        head_dim: int,
    ) -> torch.Tensor:
        tokens = int(keys.shape[1])
        if tokens <= 0:
            raise ValueError("attention requires at least one KV token")
        if head_dim > 256:
            raise ValueError(
                "fused single-token attention supports head_dim <= 256"
            )
        token_bucket = max(16, triton.next_power_of_2(tokens))
        block_d = triton.next_power_of_2(head_dim)
        block_t = int(
            os.environ.get("THINTENSOR_ATTENTION_BLOCK_T", "16")
        )
        num_warps = int(
            os.environ.get("THINTENSOR_ATTENTION_WARPS", "4")
        )
        _single_token_gqa_attention_kernel[(heads,)](
            query,
            keys,
            values,
            out,
            tokens,
            int(keys.stride(0)),
            int(keys.stride(1)),
            int(keys.stride(2)),
            int(values.stride(0)),
            int(values.stride(1)),
            int(values.stride(2)),
            heads=heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            scale=head_dim**-0.5,
            token_bucket=token_bucket,
            block_t=block_t,
            block_d=block_d,
            num_warps=num_warps,
        )
        return out

    def split_single_token_gqa_attention(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        out: torch.Tensor,
        heads: int,
        kv_heads: int,
        head_dim: int,
    ) -> torch.Tensor:
        tokens = int(keys.shape[1])
        if tokens <= 128:
            return self.single_token_gqa_attention(
                query,
                keys,
                values,
                out,
                heads,
                kv_heads,
                head_dim,
            )
        token_bucket = max(16, triton.next_power_of_2(tokens))
        splits = min(8, max(2, token_bucket // 128))
        partial_count = heads * splits
        partial_values = partial_count * head_dim
        if (
            self._attention_partial_max is None
            or self._attention_partial_max.numel() < partial_count
        ):
            self._attention_partial_max = torch.empty(
                partial_count,
                device=self.device,
                dtype=torch.float32,
            )
            self._attention_partial_den = torch.empty_like(
                self._attention_partial_max
            )
        if (
            self._attention_partial_acc is None
            or self._attention_partial_acc.numel() < partial_values
        ):
            self._attention_partial_acc = torch.empty(
                partial_values,
                device=self.device,
                dtype=torch.float32,
            )
        assert self._attention_partial_den is not None
        block_d = triton.next_power_of_2(head_dim)
        _split_gqa_attention_partial_kernel[(heads, splits)](
            query,
            keys,
            values,
            self._attention_partial_max,
            self._attention_partial_den,
            self._attention_partial_acc,
            tokens,
            int(keys.stride(0)),
            int(keys.stride(1)),
            int(keys.stride(2)),
            int(values.stride(0)),
            int(values.stride(1)),
            int(values.stride(2)),
            heads=heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            scale=head_dim**-0.5,
            token_bucket=token_bucket,
            SPLITS=splits,
            BLOCK_T=16,
            BLOCK_D=block_d,
            num_warps=4,
        )
        _split_gqa_attention_reduce_kernel[(heads,)](
            self._attention_partial_max,
            self._attention_partial_den,
            self._attention_partial_acc,
            out,
            heads=heads,
            head_dim=head_dim,
            SPLITS=splits,
            BLOCK_D=block_d,
            num_warps=4,
        )
        return out

    def matvec_argmax_tensor(
        self,
        weight: torch.Tensor,
        x: torch.Tensor,
        *,
        block_m: int | None = None,
        num_warps: int | None = None,
        config_name: str | None = None,
    ) -> torch.Tensor:
        """
        Fast default greedy lm_head path:
            Triton matvec -> torch.argmax(logits)

        This avoids the slow custom two-stage reduction unless explicitly enabled.
        Returned token is a GPU scalar. No .item() here.
        """
        if self.argmax_mode == "triton_two_stage":
            return self._matvec_argmax_tensor_triton_two_stage(weight, x)

        rows = int(weight.shape[0])
        logits = self.logits_buffer(rows, x.dtype)
        self.matvec(
            weight,
            x,
            logits,
            block_m=block_m,
            num_warps=num_warps,
            config_name=config_name,
        )

        try:
            return torch.argmax(logits)
        except RuntimeError:
            return torch.argmax(logits.float())

    def indexed_matvec_argmax_tensor(
        self,
        weight: torch.Tensor,
        x: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        if indices.ndim != 1 or indices.numel() <= 0:
            raise ValueError("indexed argmax requires a non-empty ID vector")
        rows = int(weight.shape[0])
        cols = int(weight.shape[1])
        candidate_count = int(indices.numel())
        block_k = triton.next_power_of_2(candidate_count)
        block_n = triton.next_power_of_2(cols)
        _indexed_matvec_argmax_kernel[(1,)](
            weight,
            x,
            indices,
            self._argmax_out_idx,
            rows,
            cols,
            candidate_count,
            int(weight.stride(0)),
            int(weight.stride(1)),
            BLOCK_K=block_k,
            BLOCK_N=block_n,
            num_warps=8,
        )
        return self._argmax_out_idx

    def sparse_residual_matvec(
        self,
        residual_values: torch.Tensor,
        residual_indices: torch.Tensor,
        x: torch.Tensor,
        out: torch.Tensor,
    ) -> torch.Tensor:
        if residual_values.shape != residual_indices.shape:
            raise ValueError("sparse residual values/indices shape mismatch")
        rows = int(residual_values.shape[0])
        terms = int(residual_values.shape[1])
        block_m = 64
        _sparse_residual_matvec_kernel[(triton.cdiv(rows, block_m),)](
            residual_values,
            residual_indices,
            x,
            out,
            rows=rows,
            terms=terms,
            stride_vm=int(residual_values.stride(0)),
            stride_vk=int(residual_values.stride(1)),
            stride_im=int(residual_indices.stride(0)),
            stride_ik=int(residual_indices.stride(1)),
            BLOCK_M=block_m,
            BLOCK_K=triton.next_power_of_2(terms),
            num_warps=4,
        )
        return out

    def binary_residual_matvec(
        self,
        packed_signs: torch.Tensor,
        scales: torch.Tensor,
        x: torch.Tensor,
        out: torch.Tensor,
    ) -> torch.Tensor:
        rows = int(packed_signs.shape[0])
        cols = int(x.numel())
        if tuple(packed_signs.shape) != (rows, cols // 8):
            raise ValueError("binary residual sign plane shape mismatch")
        if tuple(scales.shape) != (rows, cols // 32):
            raise ValueError("binary residual scale shape mismatch")
        block_m = 8
        _binary_residual_matvec_kernel[(triton.cdiv(rows, block_m),)](
            packed_signs,
            scales,
            x,
            out,
            rows=rows,
            cols=cols,
            stride_bm=int(packed_signs.stride(0)),
            stride_sm=int(scales.stride(0)),
            BLOCK_M=block_m,
            BLOCK_K=256,
            num_warps=4,
        )
        return out

    def selected_scaled_matvec(
        self,
        weight: torch.Tensor,
        scales: torch.Tensor,
        row_indices: torch.Tensor,
        x: torch.Tensor,
        out: torch.Tensor,
    ) -> torch.Tensor:
        rows, cols = int(weight.shape[0]), int(weight.shape[1])
        if tuple(scales.shape) != (rows,):
            raise ValueError("selected-row scales shape mismatch")
        if tuple(row_indices.shape) != (rows,):
            raise ValueError("selected-row index shape mismatch")
        block_m = 8
        _selected_scaled_matvec_kernel[(triton.cdiv(rows, block_m),)](
            weight,
            scales,
            row_indices,
            x,
            out,
            rows=rows,
            cols=cols,
            stride_wm=int(weight.stride(0)),
            BLOCK_M=block_m,
            BLOCK_K=256,
            num_warps=4,
        )
        return out

    def persistent_vocab_block_matvec_argmax_tensor(
        self,
        weight: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        rows = int(weight.shape[0])
        cols = int(weight.shape[1])
        block_n = triton.next_power_of_2(cols)

        num_programs = 64
        block_m = 32

        if (
            self._persistent_argmax_partial_vals is None
            or self._persistent_argmax_partial_vals.numel() < num_programs
        ):
            self._persistent_argmax_partial_vals = torch.empty(
                num_programs,
                device=self.device,
                dtype=torch.float32,
            )
            self._persistent_argmax_partial_idxs = torch.empty(
                num_programs,
                device=self.device,
                dtype=torch.int64,
            )

        self._persistent_argmax_counter.fill_(0)

        _persistent_vocab_block_argmax_kernel[(num_programs,)](
            weight,
            x,
            self._argmax_out_idx,
            self._argmax_out_val,
            self._persistent_argmax_counter,
            self._persistent_argmax_partial_vals,
            self._persistent_argmax_partial_idxs,
            rows,
            cols,
            int(weight.stride(0)),
            int(weight.stride(1)),
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            NUM_PROGRAMS=num_programs,
            num_warps=8,
        )
        return self._argmax_out_idx

    def _matvec_argmax_tensor_triton_two_stage(
        self,
        weight: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        rows = int(weight.shape[0])
        cols = int(weight.shape[1])
        block_n = triton.next_power_of_2(cols)
        if rows >= 65536 and cols >= 1024:
            block_m, num_warps = 32, 8
        else:
            block_m, num_warps = select_matvec_config(rows, cols)
        blocks = triton.cdiv(rows, block_m)

        if (
            self._argmax_partial_vals is None
            or self._argmax_partial_vals.numel() < blocks
        ):
            self._argmax_partial_vals = torch.empty(
                blocks,
                device=self.device,
                dtype=torch.float32,
            )
            self._argmax_partial_idxs = torch.empty(
                blocks,
                device=self.device,
                dtype=torch.int32,
            )

        assert self._argmax_partial_idxs is not None

        stage1 = (
            _matvec_argmax_stage1_nomask_n_kernel
            if cols == block_n
            else _matvec_argmax_stage1_kernel
        )

        stage1[(blocks,)](
            weight,
            x,
            self._argmax_partial_vals,
            self._argmax_partial_idxs,
            rows,
            cols,
            int(weight.stride(0)),
            int(weight.stride(1)),
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            num_warps=num_warps,
        )

        _argmax_stage2_kernel[(1,)](
            self._argmax_partial_vals,
            self._argmax_partial_idxs,
            self._argmax_out_idx,
            self._argmax_out_val,
            blocks,
            BLOCK_N=triton.next_power_of_2(blocks),
            num_warps=8,
        )

        return self._argmax_out_idx

    def matvec_argmax(self, weight: torch.Tensor, x: torch.Tensor) -> int:
        raise RuntimeError(
            "matvec_argmax() synchronizes CUDA; use matvec_argmax_tensor()"
        )

    def rms_norm(self, x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor, eps: float) -> torch.Tensor:
        n = x.numel()
        _rms_norm_kernel[(1,)](
            x,
            weight,
            out,
            n,
            eps,
            BLOCK_N=triton.next_power_of_2(n),
            num_warps=8,
        )
        return out

    def add_rms_norm(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        residual_out: torch.Tensor,
        weight: torch.Tensor,
        norm_out: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n = left.numel()
        _add_rms_norm_kernel[(1,)](
            left,
            right,
            residual_out,
            weight,
            norm_out,
            n,
            eps,
            BLOCK_N=triton.next_power_of_2(n),
            num_warps=8,
        )
        return residual_out, norm_out

    def head_rms_norm(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        out: torch.Tensor,
        heads: int,
        head_dim: int,
        eps: float,
    ) -> torch.Tensor:
        _head_rms_norm_kernel[(heads,)](
            x,
            weight,
            out,
            head_dim,
            eps,
            BLOCK_N=triton.next_power_of_2(head_dim),
            num_warps=1,
        )
        return out

    def rope_qk_inplace(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        heads: int,
        kv_heads: int,
        head_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if head_dim % 2:
            raise ValueError(f"RoPE head_dim must be even, got {head_dim}")
        half = head_dim // 2
        _rope_qk_inplace_kernel[(max(heads, kv_heads),)](
            q,
            k,
            cos,
            sin,
            heads=heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            HALF=half,
            BLOCK_HALF=triton.next_power_of_2(half),
            num_warps=1,
        )
        return q, k

    def silu_mul(self, gate: torch.Tensor, up: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        n = gate.numel()
        block = 1024
        _silu_mul_kernel[(triton.cdiv(n, block),)](
            gate,
            up,
            out,
            n,
            BLOCK_N=block,
            num_warps=4,
        )
        return out

    def add(self, left: torch.Tensor, right: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        n = left.numel()
        block = 1024
        _add_kernel[(triton.cdiv(n, block),)](
            left,
            right,
            out,
            n,
            BLOCK_N=block,
            num_warps=4,
        )
        return out

    def repeat_kv(
        self,
        v: torch.Tensor,
        out: torch.Tensor,
        heads: int,
        kv_heads: int,
        head_dim: int,
    ) -> torch.Tensor:
        block = 1024
        _repeat_kv_kernel[(math.ceil((heads * head_dim) / block),)](
            v,
            out,
            heads,
            kv_heads,
            head_dim,
            BLOCK_N=block,
            num_warps=4,
        )
        return out
