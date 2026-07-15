#!/usr/bin/env python3
"""Verify a pinned sharded SafeTensors checkpoint without loading its tensors."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import time
from pathlib import Path
from typing import Any

from safetensors import safe_open


DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-dir", required=True, type=Path)
    parser.add_argument("--target-model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hf_dir = args.hf_dir.resolve()
    target = json.loads(args.target_model.read_text(encoding="utf-8"))
    index_path = hf_dir / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index["weight_map"]
    expected_shards = {
        item["name"]: item for item in target["safetensors"]["shards"]
    }
    observed_names: set[str] = set()
    tensor_records: list[dict[str, Any]] = []
    shard_records = []
    total_file_bytes = 0
    total_tensor_bytes = 0

    for shard_name in sorted(expected_shards):
        expected = expected_shards[shard_name]
        path = hf_dir / shard_name
        actual_size = path.stat().st_size
        if actual_size != int(expected["bytes"]):
            raise ValueError(
                f"{shard_name}: size {actual_size}, expected {expected['bytes']}"
            )
        actual_sha256 = sha256(path)
        if actual_sha256 != expected["sha256"]:
            raise ValueError(
                f"{shard_name}: sha256 {actual_sha256}, expected {expected['sha256']}"
            )
        header, data_start = read_header(path)
        header_names = set(header) - {"__metadata__"}
        with safe_open(path, framework="pt", device="cpu") as handle:
            safe_names = set(handle.keys())
        if safe_names != header_names:
            raise ValueError(f"{shard_name}: SafeTensors key/header mismatch")
        expected_names = {
            name for name, mapped_shard in weight_map.items()
            if mapped_shard == shard_name
        }
        if header_names != expected_names:
            missing = sorted(expected_names - header_names)
            extra = sorted(header_names - expected_names)
            raise ValueError(
                f"{shard_name}: index/header mismatch missing={missing[:5]} extra={extra[:5]}"
            )
        ranges = []
        shard_tensor_bytes = 0
        for name in sorted(header_names):
            if name in observed_names:
                raise ValueError(f"tensor {name} occurs in more than one shard")
            observed_names.add(name)
            entry = header[name]
            dtype = str(entry["dtype"])
            shape = [int(dim) for dim in entry["shape"]]
            start, end = map(int, entry["data_offsets"])
            if start < 0 or end <= start or data_start + end > actual_size:
                raise ValueError(f"{shard_name}:{name}: invalid data offsets {start}:{end}")
            expected_bytes = element_count(shape) * DTYPE_BYTES[dtype]
            if end - start != expected_bytes:
                raise ValueError(
                    f"{shard_name}:{name}: data range {end-start}, expected {expected_bytes}"
                )
            ranges.append((start, end, name))
            shard_tensor_bytes += expected_bytes
            tensor_records.append(
                {
                    "name": name,
                    "shard": shard_name,
                    "dtype": dtype,
                    "shape": shape,
                    "data_bytes": expected_bytes,
                }
            )
        ranges.sort()
        for left, right in zip(ranges, ranges[1:]):
            if left[1] > right[0]:
                raise ValueError(
                    f"{shard_name}: tensors {left[2]} and {right[2]} overlap"
                )
        shard_records.append(
            {
                "name": shard_name,
                "bytes": actual_size,
                "sha256": actual_sha256,
                "header_bytes": data_start,
                "tensor_count": len(header_names),
                "tensor_bytes": shard_tensor_bytes,
            }
        )
        total_file_bytes += actual_size
        total_tensor_bytes += shard_tensor_bytes
        print(f"verified {shard_name}: {actual_size} bytes, {len(header_names)} tensors")

    if observed_names != set(weight_map):
        raise ValueError("aggregate SafeTensors names do not match index weight_map")
    expected_file_bytes = int(target["safetensors"]["download_file_bytes"])
    if total_file_bytes != expected_file_bytes:
        raise ValueError(
            f"total file bytes {total_file_bytes}, expected {expected_file_bytes}"
        )
    expected_tensor_bytes = int(index.get("metadata", {}).get("total_size", -1))
    if total_tensor_bytes != expected_tensor_bytes:
        raise ValueError(
            f"total tensor bytes {total_tensor_bytes}, expected {expected_tensor_bytes}"
        )

    report = {
        "schema": "thintensor.safetensors_checkpoint_verification.v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "hf_dir": str(hf_dir),
        "repo": target["resolved_hf_repo"],
        "revision": target["revision_sha"],
        "index": str(index_path),
        "index_sha256": sha256(index_path),
        "shard_count": len(shard_records),
        "tensor_count": len(tensor_records),
        "total_file_bytes": total_file_bytes,
        "total_tensor_bytes": total_tensor_bytes,
        "all_sizes_match": True,
        "all_sha256_match": True,
        "all_headers_open": True,
        "all_index_names_and_shards_match": True,
        "all_shapes_and_data_ranges_match": True,
        "shards": shard_records,
        "tensors": tensor_records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.out}")


def read_header(path: Path) -> tuple[dict[str, Any], int]:
    with path.open("rb") as handle:
        raw_length = handle.read(8)
        if len(raw_length) != 8:
            raise ValueError(f"{path.name}: truncated SafeTensors length")
        header_length = struct.unpack("<Q", raw_length)[0]
        if header_length <= 2 or header_length > path.stat().st_size - 8:
            raise ValueError(f"{path.name}: invalid header length {header_length}")
        header_bytes = handle.read(header_length)
        if len(header_bytes) != header_length:
            raise ValueError(f"{path.name}: truncated SafeTensors header")
    return json.loads(header_bytes), 8 + header_length


def element_count(shape: list[int]) -> int:
    result = 1
    for dim in shape:
        if dim < 0:
            raise ValueError(f"negative tensor dimension {dim}")
        result *= dim
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
