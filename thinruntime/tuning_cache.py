"""Persistent, GPU-scoped kernel tuning decisions.

Built-in entries are evidence-backed seeds, not generic shape assumptions.
Unknown shapes are measured by the runtime and written to the user's cache.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1

# Retained Blackwell choices from warmed real-decode measurements. These only
# apply to compute capability 12.0 and exact shape/dtype/scaling matches.
BUILTIN_MATVEC_CHOICES: dict[tuple[Any, ...], str] = {
    (12, 0, 128256, 2048, "torch.bfloat16", False): "triton",
    (12, 0, 128256, 2048, "torch.float8_e4m3fn", True): "triton",
    (12, 0, 11008, 2048, "torch.bfloat16", False): "triton",
    (12, 0, 11008, 2048, "torch.float8_e4m3fn", True): "triton",
    (12, 0, 2048, 11008, "torch.bfloat16", False): "triton_loop_256",
    (12, 0, 2048, 11008, "torch.float8_e4m3fn", True): "triton_loop_128",
    (12, 0, 2048, 2048, "torch.bfloat16", False): "triton",
    (12, 0, 512, 2048, "torch.bfloat16", False): "triton",
}


def matvec_key(
    *,
    capability: tuple[int, int],
    rows: int,
    cols: int,
    dtype: str,
    scaled: bool,
    stride: tuple[int, int],
) -> str:
    return "|".join(
        (
            f"cc={capability[0]}.{capability[1]}",
            f"shape={rows}x{cols}",
            f"dtype={dtype}",
            f"scaled={int(scaled)}",
            f"stride={stride[0]},{stride[1]}",
        )
    )


def builtin_matvec_choice(
    capability: tuple[int, int],
    rows: int,
    cols: int,
    dtype: str,
    scaled: bool,
) -> str | None:
    return BUILTIN_MATVEC_CHOICES.get(
        (capability[0], capability[1], rows, cols, dtype, scaled)
    )


class TuningCache:
    def __init__(self, gpu_fingerprint: str) -> None:
        from .model_cache import cache_root

        safe = "".join(
            char if char.isalnum() or char in "._-" else "_"
            for char in gpu_fingerprint
        )
        self.path = cache_root() / "tuning" / f"{safe}.json"
        self.data = self._load()

    def _load(self) -> dict[str, Any]:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if payload.get("schema_version") == SCHEMA_VERSION:
                return payload
        except (OSError, json.JSONDecodeError):
            pass
        return {"schema_version": SCHEMA_VERSION, "matvec": {}}

    def get_matvec(self, key: str) -> str | None:
        value = self.data.get("matvec", {}).get(key)
        return str(value["choice"]) if isinstance(value, dict) else None

    def put_matvec(
        self,
        key: str,
        *,
        choice: str,
        microbench_ms: dict[str, float],
        decode_ms: dict[str, float],
    ) -> None:
        self.data.setdefault("matvec", {})[key] = {
            "choice": choice,
            "microbench_ms": microbench_ms,
            "decode_ms": decode_ms,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(
            f".tmp-{os.getpid()}"
        )
        temporary.write_text(
            json.dumps(self.data, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.path)
