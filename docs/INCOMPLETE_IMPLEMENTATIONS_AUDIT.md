# Incomplete implementation audit

Audit date: 2026-07-14

Scope: Rust archive/converter/planner code, `thinruntime/`, installed CLI routing, runtime backend scripts, correctness tools, and benchmark tools. Generated benchmark/correctness JSON and third-party build output were excluded from source-marker searches.

Searches performed:

- `TODO`, `FIXME`, `XXX`, `HACK`, `STUB`, `NotImplementedError`, `pass`, placeholder/dummy, xfail/skip
- constant empty/zero/`None` returns
- ignored exceptions and disabled branches
- CLI argument declaration through subprocess forwarding
- manifest/page-table fields through runtime consumers
- model-family defaults and architecture fallthrough
- conversion journal, destructive deletion, verification, and free-space paths
- quantized shape/divisibility checks and benchmark accounting

## Critical-path findings

| File and line | Intended behavior | Reachable | Priority | Status and validation |
|---|---|:---:|:---:|---|
| `src/convert_hf.rs` conversion entry | Convert without a second model-sized temporary and resume after source consumption | yes | critical | Implemented deterministic partial archive, plan/journal, per-page reserve gate, fsync/read-back, guarded deletion, and resume. Forced SIGKILL test resumed after shard 1 was already absent and final verification passed. |
| `src/archive.rs` resumable writer | Reopen only a committed prefix and reject journal/page mismatches | yes | critical | Implemented deterministic prefix, committed-tail truncation, page identity/size/checksum checks, read-back verification, atomic final rename, and directory sync. |
| `src/convert_hf.rs` memory plan | Streaming minimum must not exceed the all-resident upper bound | yes | high | Fixed separately deduplicated global/stage aliases by bounding the streaming set at the all-resident unique set. A tiny tied/zero fixture previously produced an invalid manifest; public dry-run and normal conversion now pass. |
| `src/archive.rs`, `src/convert_hf.rs`, `src/main.rs`, `thinruntime/cli.py`, `thinruntime/command_runner.py` | Conversion dry-run must report exact output, temporary bound, deletion points, and projected reserve without mutation | yes | high | Implemented and public-CLI tested. A 12-page fixture reported exact output/deletion point and wrote no archive, partial, plan, or journal. |
| `thinruntime/archive.py` parser | Runtime must reject malformed layout even when full checksum verification is intentionally skipped at open | yes | critical | Fixed orphan physical page acceptance; added fixed header/manifest/table boundaries, exact `data_off`, unsupported-flag rejection, stored/raw equality, fused offset/size bounds, duplicates, ranges, and overlap checks. Strict parser and normal archive regression passed. |
| `thinruntime/model_arch.py`, `thinruntime/capabilities.py` cross-attention parsing | Invalid layer IDs must not silently alter compacted layer numbering | yes | high | Replaced ignored conversion errors with explicit `ValueError`; valid integer/string IDs and invalid values tested in both paths. |
| `thinruntime/operator_registry.py` graph selection | Missing operator contracts must not fall through to RMSNorm/GQA/gated MLP | yes | critical | Removed family defaults. Missing normalization, sequence, or feed-forward contracts now fail explicitly; valid declared contract regression passed. |
| `scripts/inspect_hf_compat.py` multimodal name projection | Pre-conversion compatibility must inspect the canonical text graph the converter will write | yes | high | Added converter-equivalent `model.language_model` projection, root LM-head retention, adapter/vision exclusion, cross-layer compaction, and source/canonical counts. The official Qwen3.6 index projects 1,199 tensors to 851 text tensors and compiles with no gaps. Native readiness is now false whenever schema compilation fails. |
| `thinruntime/gpu_runtime.py` profiler result aggregation | Unrecorded CUDA events must not appear as zero time | yes | high | Replaced swallowed event errors with `null` component values and a labeled `timing_unavailable` map. Fused-attention fixture exposed and validated the disclosure. |
| `scripts/thin_runtime.py`, `thinruntime/cli.py` benchmark provenance | Public sustained decode must accept an identical tokenized prompt, retain outputs, and state sequence/batching semantics | yes | critical | Added explicit prompt-token prefill, device-side generated-ID capture with one post-timing copy, prompt-aware positions, single-sequence/no-aggregate fields, and public/auto-search forwarding. A CUDA causal fixture retained all six prompt/decode KV positions and all generated IDs. |
| `thinruntime/gpu_runtime.py`, `scripts/thin_runtime.py` hardware telemetry | Available temperature, power, device-memory, and utilization samples must not be omitted from benchmark JSON | yes | medium | Extended the existing 0.5-second thermal guard query to record peak power, NVIDIA-reported memory, utilization, and sample count without adding another polling process. Live RTX 5050 parsing regression passed. |
| `scripts/thin_runtime.py` FP8 selector telemetry | Inactive decoder FP8 routes must not be reported as applying to `"all"` layers | yes | high | Fixed effective selector reporting: head-only FP8 now emits `null` for decoder/down/QKV/O layer selectors; active routes retain specific, generic, then `"all"` precedence. Five-case regression passed. |
| `thinruntime/gpu_runtime.py` `resident_weight_bytes` | Page-pool registered external head must be counted once | yes | high | Fixed double counting and added a 100-byte pool/20-byte registered-head regression. |
| `thinruntime/memory_report.py`, `scripts/thin_runtime.py` | Report non-overlapping memory ownership and allocator peaks | yes | high | Implemented page weights, registered external allocations, retired tensors, expert cache, sidecars, scratch, GPU/CPU KV, CPU/pinned cache, packed CPU INT4 weights/scales, allocator allocation/reserve, RSS, and mmap virtual bytes. A late target-report audit caught and fixed the previously implicit 15.22-GB INT4 owner; synthetic 280-byte ownership total passed. Integrated into runtime benchmark JSON. |
| `thinruntime/gpu_runtime.py` bytes/token property | Specialized recurrent/Gemma graphs without `LayerPlan` must not report zero body traffic | yes | high | Replaced `LayerPlan`-only logic with executable page accounting, including layer scales, final norm, head, and sidecars while excluding duplicate fused physical containers. Synthetic accounting regression passed. |
| `thinruntime/gpu_runtime.py`, `scripts/thin_runtime.py`, `thinruntime/cli.py` dense Q2 route | A dense 27B body needs a sub-MXFP4 resident representation | yes, opt-in | experimental/high | Implemented MSE-selected Q1/Q2 E8M0 groups, bounded CUDA packing, page-pool sizing/loading, one-expert tensor-core dispatch, derived cache, CPU-cache release, flags, telemetry, and auto-search candidate. Kernel cosine versus explicit dequantized matvec was `0.99999857`; converted causal fixture and 7/7 cache-hit rerun passed. Target acceptance remains gated on the real model. |
| `scripts/dump_hf_sequential_logits.py` | BF16 reference must fit bounded RAM and use official model math | yes | critical | Implemented one official decoder layer at a time, sliced embedding rows, eager causal mask, and chunked head. Synthetic Qwen3.5 logits were bit-identical to the fully resident official text model. |
| `scripts/verify_safetensors_checkpoint.py` | Preserve sizes, hashes, headers, tensor names, shapes, dtypes, ranges, and index mapping before source deletion | yes | critical | Implemented bounded header verification and synthetic BF16/F32 test. The exact target passed all 15 pinned hashes, 1,199 indexed tensors, shard mappings, dtypes, shapes, and non-overlapping ranges before consumption. |
| `scripts/compare_thin_modes.py`, `scripts/compare_saved_logits.py`, `scripts/compare_llamacpp_q8_logits.py` | Compare full vocabulary on one fixed teacher trajectory without treating log-softmax offsets as quality loss | yes | critical | Added saved NPZ vectors, per-candidate array prefixes, Q4 log-probability handling, centered cosine min/mean/percentiles, centered errors, top-1 rate/first divergence, top-5 overlap/exact order, JS, and TV. Self-comparison regression passed; exact target comparison remains benchmark-gated. |
| `thinruntime/gpu_runtime.py`, `scripts/thin_runtime.py`, `thinruntime/cli.py` calibrated CPU INT4 | Execute a quality-preserving 27B body without hiding host residency or one-time packing | yes, opt-in | high | Added affine group-32 fitting, block-GPTQ covariance feedback, CPU AVX-512 execution, atomic checksum/calibration-keyed caches, public CLI flags, and host/kernel/transfer telemetry. Exact target vectors beat Q4 centered cosine; 200-token speed is retained as a failed target result. |
| `thinruntime/triton_kernels.py` affine grouped INT4 | Move calibrated pages to GPU without changing their codes or using the slow replicated-dot route | yes, opt-in | high | Added affine offsets to the bandwidth grouped kernel and projection selectors. Real-page kernel check reached cosine 0.9999854 versus CPU and 317.2 GB/s; full fixed-teacher logits passed. Public diagnostic remains slower than Q4 end to end. |

## Marker and fallback inventory

These are deliberate fallbacks, not unfinished execution paths:

| File and line | Behavior | Reachable | Priority | Final decision / test |
|---|---|:---:|:---:|---|
| `thinruntime/cli.py:31-38` | Rich UI import is optional | yes | low | Keep plain-text fallback; CLI help/version tested without depending on Rich behavior. |
| `thinruntime/hf_loader.py:11-18`, `thinruntime/gpu_runtime.py:28-35`, `scripts/run_thin_torch.py:25-32`, `scripts/compare_thin_vs_hf_runtime.py:54-61` | `malloc_trim` is best-effort on glibc | yes | low | Keep; allocator correctness does not depend on it and memory report discloses observed RSS/CUDA state. |
| `thinruntime/tuning_cache.py:74-81` | Missing/corrupt tuning JSON falls back to an empty versioned cache | yes | low | Keep; corrupt cache cannot affect numerical correctness and choices are remeasured. |
| `thinruntime/gpu_runtime.py:961-977`, `thinruntime/gpu_runtime.py:12128-12143` | Optional tensor lookup caches miss into the authoritative registry | yes | low | Keep; `KeyError` is the normal cache-miss signal and missing required tensors still raise explicitly. |
| `thinruntime/gpu_runtime.py` derived-cache unlink/write cleanup blocks | Corrupt cache is discarded; failed cleanup does not hide recomputation | yes | low | Keep best-effort unlink. Metadata, shape, dtype, and checksum-key validation precede every hit. |
| `thinruntime/gpu_runtime.py` CPU-embedding close fallback | PyTorch may briefly retain an mmap buffer export at isolated-process exit | yes | medium | Keep narrowly scoped `BufferError` suppression only when CPU embedding is enabled; non-CPU-embedding close errors still raise. Process exit releases mapping/fd. |
| `scripts/compare_hf_thin_logits.py:185-213` | Optional Transformers/kernels probing selects bounded CPU/offload reference | yes | low | Keep conservative fallback; it changes placement only, not reference math. |
| `thinruntime/cli.py` optional package/CUDA/cache discovery exception blocks | Missing metadata or device probing selects explicit conservative candidates | yes | low | Keep; selected command and capability data are emitted in benchmark JSON. |

No source `TODO`, `FIXME`, `XXX`, `HACK`, `STUB`, `NotImplementedError`, skipped test, or xfail remains in a critical path. Experimental mechanisms remain default-off and are labeled; none is silently selected as a safe profile without isolated auto-search success.

## Remaining completion gates

- The exact checkpoint, official reference, consuming conversion, archive verification, exact smoke, three-way logits, and public 200-token ThinTensor run are complete.
- The restored exact Q4_K_M completed matched exact-prompt 200/500-token evidence and was hash-reverified and deleted. The accepted ThinTensor profile also completed a matched 500-token run with 512/512 retained causal KV positions.
- Rejected derived-cache identities were removed; 496 accepted checksum/calibration-keyed pages remain for reproduction. Final free space is 43,144,323,072 bytes.
- No implementation or evidence gate remains. The final report explicitly records that cosine passed while matched long-run speed failed by 53.25%.
