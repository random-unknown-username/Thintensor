You are working in this repo:

`/home/satvik/Projects/thintensor-opus`

Project: ThinTensor, a `.thin` archive + GPU runtime for LLM inference. The goal is to make `.thin` files easy to read, stream, offload, plan, and execute faster than normal BF16 runtimes on limited VRAM.

Current important benchmark result:

* HF Transformers BF16 failed to load Qwen3-4B-Instruct-2507 on an 8GB GPU.
* HF error: CUDA OOM at around 7.96GB peak allocated.
* ThinTensor all-resident BF16 also OOMs for the same 4B model, as expected.
* ThinTensor streaming mode starts running but currently crashes due to runtime architecture bugs.
* On Qwen3-0.6B, ThinTensor BF16 greedy decode was ~173 tok/s.
* On Qwen3-0.6B, ThinTensor BF16 body + FP8 LM head reached ~204 tok/s.
* That means selective FP8 LM head is a real optimization and should be preserved.
* However, current FP8 LM head mode appears additive for 0.6B: it keeps BF16 lm_head and adds FP8 lm_head. For 4B, that is not viable. FP8 lm_head must be able to replace BF16 lm_head residency.

Hard rules:

* Do NOT use `.item()`, `.cpu()`, `.tolist()`, `float(...)`, `int(...)`, `topk()`, or `torch.cuda.synchronize()` inside timed decode loops.
* Token IDs must stay GPU scalar tensors during decode.
* `next_token_tensor()` must return a GPU tensor and must not synchronize.
* `next_token()` may exist only as a dangerous/debug helper and should raise or loudly warn because it synchronizes.
* `runtime.topk()` must only run after timing.
* Do not fake benchmark speedups by skipping required model work.
* Do not break `.thin` archive loading.
* Do not make memory-heavy optimizations default unless benchmarks prove they are worth it.
* Do not use `apply_patch`; edit files directly with normal file writes or Python scripts.

Files to read first:

* `thinruntime/archive.py`
* `thinruntime/torch_loader.py`
* `thinruntime/gpu_runtime.py`
* `thinruntime/triton_kernels.py`
* `scripts/thin_runtime.py`
* all scripts under `scripts/`
* Rust CLI code for `./target/release/thintensor`, especially `convert-hf`, `pack`, `inspect`, `stats`, `verify`, `plan`, `profile`, `simulate-load`, `bench-load`, `bench-plan`, `bench-baseline`, `bench-archive`, and `repack`.

Current known runtime/code problems:

1. Dirty `triton_kernels.py`
   The file currently has accidental duplicate or messy methods from previous patching. Clean it.
   Look for and fix:

* duplicate `_argmax_stage2_kernel`
* duplicate `logits_buffer`
* duplicate `matvec_argmax`
* any benchmark/helper function accidentally nested inside `TritonDecodeBackend.__init__`
* any duplicate method definitions that shadow each other
* any broken indentation
* any stale comments saying the wrong mode is default
  After cleanup, `python3 -m py_compile thinruntime/triton_kernels.py` must pass.

2. `multi_matvec` crashes on Qwen3-4B
   Streaming Qwen3-4B currently crashes with:
   `ValueError: multi_matvec currently requires power-of-two columns`

Qwen3-4B hidden size is not power-of-two, so `multi_matvec` must not raise. Fix it.
Requirements:

* If columns are power-of-two and the optimized fused kernel works, use it.
* If columns are not power-of-two, fall back to normal per-weight matvec into slices of the output buffer.
* Correctness first. Speed second.
* This fallback must not allocate unnecessary temporary tensors.
* It must work for q/k/v fused multi-matvec and gate/up fused multi-matvec.
* It must return the same output buffer.

3. Qwen3-4B head_dim / KV shape bug
   After bypassing the power-of-two crash, streaming Qwen3-4B crashes with:
   `RuntimeError: shape '[12, 80]' is invalid for input of size 1024`

This means runtime inferred wrong KV dimensions:

* Runtime tried to reshape K into `kv_heads=12, head_dim=80`, total 960.
* Actual K size is 1024.
* For Qwen3-4B, actual likely KV shape is `kv_heads=8, head_dim=128`, total 1024.
* Do not assume `head_dim = q_dim / num_attention_heads` blindly.
* Some modern Qwen models have explicit `head_dim` in config or projection shapes that must be respected.

Fix head dimension inference in `ThinGpuQwenRuntime`.
Requirements:

* Read `head_dim` from manifest/model config if present.
* If absent, infer from actual projection shapes and config:

  * `q_dim = q_proj.weight.shape[0]`
  * `kv_dim = k_proj.weight.shape[0]`
  * prefer explicit `head_dim`
  * otherwise prefer `kv_dim / num_key_value_heads` if divisible
  * otherwise use a candidate list like 128, 96, 80, 64, 256, but only if both q_dim and kv_dim are divisible by it
* Set:

  * `self.head_dim`
  * `self.heads = q_dim // head_dim`
  * `self.kv_heads = kv_dim // head_dim`
* If a KV cache was constructed from stale manifest values before runtime inference, fix its `kv_heads` and `head_dim` before any KV blocks are allocated.
* If KV blocks already exist, raise a clear error rather than silently corrupting.
* Add telemetry fields:

  * `runtime_heads`
  * `runtime_kv_heads`
  * `runtime_head_dim`
  * `runtime_q_dim`
  * `runtime_kv_dim`
  * `manifest_heads`
  * `manifest_kv_heads`
  * `manifest_head_dim` if present

4. Streaming/offload weight residency
   `--residency all` OOMs for Qwen3-4B BF16 on 8GB. That is expected.
   But `--residency stream` must work.
   Fix streaming mode so it:

* Does not keep all layer weights resident.
* Keeps only persistent tensors needed globally, such as embeddings and maybe lm_head.
* Loads current layer weights, uses them, and evicts completed layer weights.
* Supports `--prefetch 0` safely.
* Supports `--prefetch 1` only if it does not OOM.
* Reports:

  * resident pages
  * resident bytes
  * peak resident bytes
  * GPU transfer bytes
  * CPU staging bytes
  * evicted pages
  * page faults
  * persistent pages
* Does not keep accidental references to old layer tensors through layer plans that prevent eviction.

Important: static layer plans must not break streaming.
If direct tensor refs are cached in the layer plan, they may keep streamed layer pages alive forever. For streaming mode, either:

* store page IDs and resolve tensors per layer, then release/evict; or
* make the plan aware of streaming and not keep strong tensor refs.
  For all-resident mode, direct tensor refs are okay.

5. Static layer plan
   For all-resident mode, optimize runtime by building a per-layer plan that avoids repeated string formatting and dict lookup in the decode loop.
   Each plan may hold direct tensor refs for all-resident mode:

* input_layernorm
* q_proj
* k_proj
* v_proj
* o_proj
* q_norm if present
* k_norm if present
* post_attention_layernorm
* gate_proj
* up_proj
* down_proj

For streaming mode, layer plan must not keep GPU tensor refs that prevent eviction. It can hold IDs and resolve current page tensors on demand.

6. Runtime fusion
   Runtime fusion of q/k/v and gate/up doubled VRAM for tiny speedup on Qwen3-0.6B. Do not enable by default.
   Keep it opt-in:
   `THINTENSOR_RUNTIME_FUSE=1`

If enabled:

* q/k/v can be concatenated into one GPU matrix per layer.
* gate/up can be concatenated into one GPU matrix per layer.
* Must report `runtime_fusion_enabled` and `runtime_fusion_extra_bytes`.
* Must not run for streaming unless explicitly safe and memory-bounded.
* If it gives tiny speedup and huge VRAM overhead, leave it disabled by default.

7. Selective FP8 LM head
   Keep and improve this. It is currently the best result.
   On Qwen3-0.6B:

* BF16 head: ~173 tok/s
* FP8 head: ~204 tok/s
* top-1 token stayed same in sample
  This should become a first-class feature.

But for 4B, FP8 head must be replacement mode, not additive mode.
Implement:

* `--lm-head-fp8` enables FP8 lm_head.
* Add option `--keep-bf16-lm-head` if exact BF16 head should remain resident too.
* By default, when `--lm-head-fp8` is enabled, do not keep BF16 lm_head resident on GPU if it can be avoided.
* If `.thin` archive loader would normally load lm_head BF16 as persistent, skip or evict it after FP8 head is built.
* Better: support archive/load-time conversion of lm_head to FP8 without ever creating both full GPU copies if possible.
* Telemetry:

  * `lm_head_fp8_enabled`
  * `lm_head_fp8_bytes`
  * `lm_head_bf16_resident`
  * `lm_head_bf16_bytes`
  * `lm_head_memory_saved_bytes`
  * `lm_head_net_extra_bytes`
* `next_token_tensor()` should use FP8 head directly when enabled.
* Top-k after timing should use the same active head unless `--exact-topk` is requested.

8. BF16 KV default
   For BF16 exact mode, default KV old codec should be `bf16`, not `q4`.
   Ensure:

* `--kv` default is `bf16` for BF16 runtime runs.
* `kv_old_codec` reports `bf16`.
* `kv_compressed_blocks` is zero in exact BF16 mode.
* If the user explicitly asks q4/q8/fp8 KV, then use that.
* KV offload policy should not corrupt exact mode.

9. Timing pollution in `scripts/thin_runtime.py`
   Fix benchmark loop.
   Current/previous issue: final iteration did `runtime.topk()` inside timed loop.
   Required decode loop:

* Every step does:

  * `hidden = runtime.forward_token(token_id, ...)`
  * `token_id = runtime.next_token_tensor(hidden)`
* No `topk` inside loop.
* No `.item()` inside loop.
* No CPU conversion inside loop.
* No synchronize inside loop except deliberate first-token measurement if separately excluded.
* After loop timing ends, compute `top_logits = runtime.topk(hidden, args.top_k)`.

Output JSON must clearly distinguish:

* `decode_s`
* `steady_decode_s`
* `tokens_per_s`
* `steady_tokens_per_s`
* `first_token_s`
* whether top_logits were computed after timing

10. LM head argmax mode
    On Qwen3-0.6B, synced microbenchmarks made Triton two-stage look good, but real pipelined `next_token_tensor` showed torch argmax path was faster.
    Therefore:

* Do not choose default based only on per-call synchronized timings.
* Benchmark in real decode loop.
* Default should be the fastest stable real decode mode.
* Keep `triton_two_stage` optional for experiments.
* Do not call `.item()` in either path.

11. `.thin` archive / convert-hf support
    The Rust binary supports:
    `./target/release/thintensor convert-hf`
    Make sure Qwen3-4B convert works from a local HF folder like:
    `Qwen3-4B/`
    containing:

* `config.json`
* `generation_config.json`
* tokenizer files
* `model-00001-of-00003.safetensors`
* `model-00002-of-00003.safetensors`
* `model-00003-of-00003.safetensors`
* `model.safetensors.index.json`

Requirements:

* Converter should preserve enough metadata for modern Qwen:

  * hidden_size
  * intermediate_size
  * num_hidden_layers
  * num_attention_heads
  * num_key_value_heads
  * head_dim if present
  * vocab_size
  * tie_word_embeddings
  * dtype
* If config has `head_dim`, include it in `.thin` manifest.
* If config does not have it, runtime must infer safely from shapes.
* Inspect/stats should show useful architecture metadata.

12. DirectGpuPageLoader / load path
    Avoid per-page CUDA synchronization.
    Requirements:

* Use pinned staging if enabled.
* Keep pinned staging tensors alive until stream sync.
* Synchronize once at end of all-resident load.
* For streaming, do not let pending copies get freed before GPU copy completes.
* Do not report fake load time that excludes required final sync.
* If OOM happens, error should include current page id, shape, dtype, attempted allocation bytes, resident bytes, and hint to use stream/FP8 head.

13. Per-shape backend choices
    Autotune currently reports choices like:

* `1024x1024 bf16: triton`
* `1024x2048 bf16: triton`
* `151936x1024 bf16: triton`
  For Qwen3-4B, new shapes likely include non-power-of-two columns like 2560.
  Implement robust per-shape backend selection:
* benchmark Triton vs `torch.mv`
* cache by `(rows, cols, dtype, stride)`
* if Triton kernel cannot support a shape, fallback to torch, not crash
* expose choices in JSON
* allow `THINTENSOR_DISABLE_AUTOTUNE=1`

14. Tests / commands to run
    Before final answer, run:

Compile checks:
`python3 -m py_compile thinruntime/triton_kernels.py thinruntime/gpu_runtime.py scripts/thin_runtime.py`

Search for sync hazards:
`rg -n "\.item\(|\.cpu\(|tolist\(|topk\(|synchronize\(|next_token\(" scripts thinruntime`

Smoke Qwen3-0.6B:
`python3 scripts/thin_runtime.py run /tmp/thintensor-first-run/qwen3-0.6b.thin --device cuda --dtype bf16 --steps 50 --residency all --kernel-backend triton-matvec --warmup-steps 2 --json`

Qwen3-0.6B FP8 head:
`python3 scripts/thin_runtime.py run /tmp/thintensor-first-run/qwen3-0.6b.thin --device cuda --dtype bf16 --steps 100 --residency all --kernel-backend triton-matvec --warmup-steps 5 --lm-head-fp8 --json`

Qwen3-4B stream smoke:
`python3 scripts/thin_runtime.py run "$THIN_4B" --device cuda --dtype bf16 --steps 5 --residency stream --kernel-backend triton-matvec --warmup-steps 0 --prefetch 0 --json`

Qwen3-4B stream real:
`python3 scripts/thin_runtime.py run "$THIN_4B" --device cuda --dtype bf16 --steps 50 --residency stream --kernel-backend triton-matvec --warmup-steps 1 --prefetch 0 --json`

Qwen3-4B stream FP8 head:
`python3 scripts/thin_runtime.py run "$THIN_4B" --device cuda --dtype bf16 --steps 50 --residency stream --kernel-backend triton-matvec --warmup-steps 1 --prefetch 0 --lm-head-fp8 --json`

15. Expected story after fixes
    We want to prove:

* HF Transformers BF16 Qwen3-4B OOMs on 8GB.
* ThinTensor can run the same Qwen3-4B `.thin` archive in streaming mode.
* ThinTensor can optionally use BF16 body + FP8 LM head to reduce decode bottleneck.
* ThinTensor reports memory honestly.
* ThinTensor does not silently use q4 KV unless requested.
* ThinTensor does not hide CPU syncs in benchmark loops.

Deliverables:

1. Cleaned code.
2. Explanation of every bug fixed.
3. Benchmark JSON before/after.
4. Exact commands run.
5. Clear memory story:

   * HF BF16 OOM peak allocated
   * ThinTensor stream peak allocated/reserved
   * ThinTensor FP8 head memory delta
6. If any optimization is not worth it, gate it behind env flag or revert it.
7. Do not claim success if Qwen3-4B does not run yet. Make the failure clear and actionable.
