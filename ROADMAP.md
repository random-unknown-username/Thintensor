# ThinTensor Roadmap

## Schema-driven generalization

- Removed the Python model-type allowlist. Common causal decoders compile from
  config fields and tensor roles.
- Added explicit architecture, attention, MLP, expert, layer-window, RoPE, and
  quantization traits to the internal descriptor and Rust manifest.
- Added a role compiler for separate/fused dense projections, packed MoE, and
  separate-expert MoE layouts.
- New archives are physically execution-ordered and carry per-page quantization
  and backend-layout metadata.
- Added native selected-expert MXFP4 Triton matvec, GPU-only router selection,
  attention sinks, alternating sliding/full causal KV, and standard RoPE
  variants.
- Added descriptor-driven compatibility inspection and automatic mode/residency
  selection.

Highest-impact remaining generalization work:

1. archive-time stacking for separate-per-expert Mixtral/Qwen-MoE checkpoints;
2. native GPTQ/AWQ/compressed-tensor Q8/Q4/Q2 codec adapters;
3. optimized Triton fused-QKV/fused-gate execution instead of the reference
   fallback;
4. learned absolute positions/Conv1D and ALiBi schema executors;
5. real large-model end-to-end benchmarks before any 30% claim.

## 2026-07-01 optimization loop

- Exact BF16 validation now pins HF to eager attention and passes the strict
  required matrix at minimum cosine 0.999316.
- Debug-only all-layer component capture and first-divergence tracing are
  available through `forward_token_debug_all()` and
  `--find-first-divergence`.
- Independent gate/up, down, QKV, and O FP8 layer masks allow sensitivity-aware
  precision placement without making body FP8 the default.
- The retained SmolLM3-3B general speed/quality configuration is gate/up FP8 in
  all layers and down FP8 in layers 8:28. It measured 46.41 tok/s over 500
  tokens with worst required-test cosine 0.99804 and 2,070,749,184 resident
  weight bytes saved. Adding O FP8 reaches about 49.88 tok/s but is demoted
  because the broader chat-template cosine falls to 0.98471.
- Role-aware kernel overrides are exposed for reproducible tuning, but no
  override beat the real-decode default by the required 2%.
- `scripts/stress_validate_thin.py` covers ordinary and adversarial prompts,
  chat templates, deterministic temperature/top-p/top-k sampling, 128-token
  generation, 512/1024-token contexts, KV growth, and peak VRAM. It reports
  teacher-forced logit parity separately from free-running generation.

Next optimization candidates require a larger architectural change rather than
another launch-parameter sweep:

1. packed native FP8 dot kernels that avoid per-row scale overhead;
2. a quality-preserving LM-head scheme with exact candidate refinement;
3. fused attention/O execution that reduces intermediate traffic without SDPA
   overhead;
4. archive-time role-specific layouts for the retained mixed-precision plan.

## Current State

- `.thin` v0 archive: fixed header, manifest JSON, page table, raw page blobs.
- Strict verifier: page table/manifest agreement, offsets, overlaps, sizes, checksums.
- HF converter: Qwen/Llama safetensors to verified `.thin`, original tensor page ids, no fusion.
- Planner: estimates weights, scratch, KV cache, total fit for backend/VRAM/context.
- Planner now supports batch size, KV dtype, residency modes, GPU fraction, JSON output, metadata overhead, and exact duplicate shared-weight savings.
- Stats/inspect: largest pages, per-layer/per-op totals, dtype/layout distributions, execution tape coverage, and JSON summaries.
- Profiles: target-VRAM sidecars with candidate plans, hot pages, streamable pages, CPU-offload candidates, KV codec recommendation, and layer/op memory costs.
- Load simulation: page load order, resident/evicted pages, sequential-read ratio, stream groups, prefetch IO ops, staging memory, peak memory, and estimated bytes moved per generated token.
- Bench probes: HF metadata load, Thin archive load, verify timing, plan timing.
- Runtime probes: HF baseline generation and ThinTensor hydrated generation.
- Repack: execution-ordered page layout preserving raw page bytes.
- Repack layouts: `execution_ordered_v1` for tape traversal and `hot_stream_v1` for contiguous hot pages plus sequential streamed layers.
- First-run pipeline: build/download-or-reuse/convert/verify/stats/bench/compare plus one automatic optimization loop.
- Benchmark reliability: first-run pipeline supports repeated subprocess trials and records median/min/max tokens/sec.
- Real sample: `Qwen/Qwen3-0.6B` converts and verifies as 311 pages.
- Native single-token runtime: static layer plans, cached per-shape backend
  choices, multi-pointer QKV/gate-up launches, direct grouped-V `o_proj`, and
  honest CUDA-event breakdown profiling.
- Fused archive aliases: `fused_decode_v1` stores QKV and gate/up once as
  physical row-concatenated pages while preserving stable logical tensor IDs.
- Opt-in FP8 LM head: BF16 layers/KV plus a separately benchmarked FP8 head
  reached 202.53 tok/s in a 500-step run with an explicit 155,582,464-byte VRAM
  cost and recorded fixed-token logit deltas.

## Next Loop

1. Add more baseline inference benchmark refinements.
   - cold-cache runs
   - longer prompts and batches
   - temperature/frequency logging

2. Build a tensor-core-native packed weight profile. Weight-only scalar
   dequantization was slower; future INT8/FP8 layer compression must quantize
   activations and use native dot instructions while preserving a BF16 control.

3. Add richer runtime simulation.
   - compare multiple profiles in one report
   - automatic layout choice from profile + simulation

4. Optional later.
   - zstd repack with clear raw-checksum semantics
   - fused QKV/gate-up physical page format
