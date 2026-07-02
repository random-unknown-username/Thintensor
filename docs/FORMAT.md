# ThinTensor v0 Format

All integers are little-endian. Strings are UTF-8. v0 stores raw page blobs only.

```text
.thin file
├── fixed binary header
├── manifest JSON bytes
├── page table records
└── raw page blobs
```

## Header

88 bytes:

| Offset | Size | Field |
| --- | ---: | --- |
| 0 | 8 | magic `THINv0\0\0` |
| 8 | 4 | `header_len`, currently `88` |
| 12 | 4 | `version`, currently `0` |
| 16 | 8 | `manifest_off` |
| 24 | 8 | `manifest_len` |
| 32 | 8 | `page_table_off` |
| 40 | 8 | `page_count` |
| 48 | 8 | `data_off` |
| 56 | 32 | `archive_hash`, reserved zero in v0 |

The header, manifest, page table, and data section are contiguous in v0.

## Manifest

The manifest is human-readable JSON:

```json
{
  "format": "thintensor",
  "version": 0,
  "model": {
    "arch": "llama",
    "hidden_size": 4096,
    "layers": 32,
    "heads": 32,
    "kv_heads": 8,
    "dtype": "q4_mixed"
  },
  "execution_tape": [
    {
      "stage": "layer_0_attn_qkv",
      "page_refs": ["L0_QKV"]
    }
  ],
  "pages": [
    {
      "id": "L0_QKV",
      "kind": "weight",
      "layer": 0,
      "op": "attn_qkv",
      "dtype": "q4",
      "shape": [12288, 4096],
      "backend_layout": "generic",
      "size": 25165824,
      "checksum": "blake3_hex_here"
    }
  ],
  "memory_plan": {
    "scratch_bytes": 67108864,
    "min_vram_bytes": 4294967296,
    "recommended_vram_bytes": 8589934592,
    "kv_cache": {
      "policy": "paged",
      "recent_tokens_high_precision": 256,
      "old_tokens_codec": "q4"
    }
  }
}
```

Known page kinds: `weight`, `embedding`, `lm_head`, and `fused_physical`.
Unknown kinds require `"experimental": true`.

Fused decode archives also support `fused_physical` pages. Logical tensor pages
retain their stable IDs and add:

```json
{
  "fused_to": "layer_0_attn_qkv_fused",
  "fused_offset": 4194304
}
```

`fused_offset` is a byte offset. A logical page has no page-table record; its
checksum covers its slice of the physical parent. The physical parent has one
page-table record and a checksum covering the full concatenated blob.

## Page Table Record

Records are variable-length and contiguous:

| Field | Type |
| --- | --- |
| `page_id_len` | `u16` |
| `page_id` | `[u8; page_id_len]` |
| `offset` | `u64` |
| `stored_size` | `u64` |
| `raw_size` | `u64` |
| `flags` | `u32` |
| `checksum` | `[u8; 32]` |

For v0:

- `flags = 0`
- `stored_size == raw_size`
- blob bytes start at `offset`
- blob bytes are uncompressed

Reserved flag bits for later:

- bit 0: zstd
- bit 1: encrypted
- bit 2: delta overlay

## Verification

The verifier rejects:

- unknown format version
- duplicate manifest page ids
- duplicate page table ids
- execution tape references to missing pages
- manifest pages missing from page table
- page table entries not present in manifest
- page offsets outside the file
- page blobs before `data_off`
- overlapping page blobs
- checksum mismatch
- declared size different from stored size
- empty, zero, or overflowing shapes
- `memory_plan.min_vram_bytes < scratch_bytes`
- unknown page kind unless experimental
