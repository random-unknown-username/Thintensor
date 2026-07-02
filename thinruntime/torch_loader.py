import math
import warnings
from functools import lru_cache
from typing import Dict

import torch

from .archive import ThinArchive


_DTYPE_MAP: Dict[str, torch.dtype] = {
    "F16": torch.float16,
    "FP16": torch.float16,
    "FLOAT16": torch.float16,
    "HALF": torch.float16,
    "BF16": torch.bfloat16,
    "BFloat16".upper(): torch.bfloat16,
    "BFLOAT16": torch.bfloat16,
    "F32": torch.float32,
    "FP32": torch.float32,
    "FLOAT32": torch.float32,
    "FLOAT": torch.float32,
    "F64": torch.float64,
    "FP64": torch.float64,
    "FLOAT64": torch.float64,
    "DOUBLE": torch.float64,
    "I64": torch.int64,
    "INT64": torch.int64,
    "I32": torch.int32,
    "INT32": torch.int32,
    "I16": torch.int16,
    "INT16": torch.int16,
    "I8": torch.int8,
    "INT8": torch.int8,
    "U8": torch.uint8,
    "UINT8": torch.uint8,
    "BOOL": torch.bool,
    "BOOLEAN": torch.bool,
    "FLOAT8_E4M3FN": torch.float8_e4m3fn,
}

_DTYPE_NBYTES: Dict[torch.dtype, int] = {
    torch.float16: 2,
    torch.bfloat16: 2,
    torch.float32: 4,
    torch.float64: 8,
    torch.int64: 8,
    torch.int32: 4,
    torch.int16: 2,
    torch.int8: 1,
    torch.uint8: 1,
    torch.bool: 1,
    torch.float8_e4m3fn: 1,
}


@lru_cache(maxsize=128)
def map_dtype(dtype_str: str) -> torch.dtype:
    key = str(dtype_str).strip().replace("torch.", "").upper()
    try:
        return _DTYPE_MAP[key]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype string: {dtype_str!r}") from exc


def dtype_nbytes(dtype: torch.dtype) -> int:
    try:
        return _DTYPE_NBYTES[dtype]
    except KeyError:
        # slow fallback for weird future torch dtypes
        return torch.empty((), dtype=dtype).element_size()


def _shape_numel(shape: tuple[int, ...]) -> int:
    if not shape:
        return 1
    if any(dim < 0 for dim in shape):
        raise ValueError(f"invalid negative dimension in shape: {shape}")
    return math.prod(shape)


def load_tensor_view(archive: ThinArchive, page_id: str) -> tuple[torch.Tensor, bool]:
    """
    Return a zero-copy CPU tensor view over a ThinTensor archive page.

    This does not copy page bytes into Python.
    The returned tensor is CPU-backed and read-only at the mmap/buffer level.
    Runtime code can then copy it to pinned memory/GPU.

    Important:
    The archive must stay open while this CPU tensor view is alive.
    """
    metadata = archive.get_tensor_metadata(page_id)

    shape = tuple(int(x) for x in metadata["shape"])
    dtype = map_dtype(str(metadata["dtype"]))
    elem_size = dtype_nbytes(dtype)

    size_bytes = int(metadata["size"])
    if size_bytes < 0:
        raise ValueError(f"{page_id}: negative byte size {size_bytes}")
    if size_bytes % elem_size != 0:
        raise ValueError(
            f"{page_id}: byte size {size_bytes} is not divisible by dtype size {elem_size}"
        )

    count = size_bytes // elem_size
    expected = _shape_numel(shape)
    if count != expected:
        raise ValueError(
            f"{page_id}: metadata mismatch: shape {shape} expects {expected} elements, "
            f"but page has {count} elements from {size_bytes} bytes of {dtype}"
        )

    if not hasattr(archive, "get_page_view"):
        raise RuntimeError(
            "ThinArchive.get_page_view(page_id) is required for zero-copy loading. "
            "Update thinruntime/archive.py first."
        )

    view = archive.get_page_view(page_id)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The given buffer is not writable")
        tensor = torch.frombuffer(view, dtype=dtype, count=count)

    return tensor.reshape(shape), True
