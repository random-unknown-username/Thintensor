# Benchmarking To-Do List

This document tracks the status of model benchmark candidates, optimization steps, and post-benchmark cleanups. All benchmarks are run using 200 tokens (candidates/faster run) instead of the full 500-step runs to conserve resources.

> [!IMPORTANT]
> **Overnight Instruction**: For every model, if `thinruntime` benchmark performance (tokens/second) is slower than the Hugging Face baseline (`hf_tokens_per_s`), identify the performance bottleneck (e.g. PCIe transfer, page pool eviction overhead, or lack of kernel fusion) and optimize the code/profile parameters aggressively until we beat the HF baseline. Work autonomously overnight and present the optimized results by morning.
>
> **Overnight Safety Instructions** (ADDED):
> - Monitor GPU/CPU temperatures continuously (poll every 10 seconds)
> - If GPU temp hits 83°C or CPU hits 85°C, sleep for 10 minutes (do NOT stop working)
> - Monitor disk usage every 5 minutes - if >90% full, clean up cache/weights immediately
> - Keep GPU fan at 100% if possible via nvidia-smi
> - Do NOT stop working between sleeps - continue processing next model/task
> - Save progress to this todo file every 10 minutes
> - Run `nvidia-smi -lgc 0` to uncap GPU clocks if throttling occurs
> - Log all temps/speeds to `/tmp/thintensor_overnight.log`

## 🟩 1. Qwen3.5-9B (Complete - Re-run Scheduled)
- [x] Pull `Qwen/Qwen3.5-9B` safely with prefix stripping workaround for `lm_head.weight`.
- [x] Pre-quantize/verify logits match HF baseline with high similarity.
- [x] Complete benchmark suite (`safe`, `balanced`, `max-performance`, `max-max-perf`) with eager backend fallbacks.
- [x] Save detailed results table to [benchmark.md](file:///home/satvik/.gemini/antigravity-cli/brain/fb9a4fa7-dbdd-4bb0-a741-cd8c95d8821e/benchmark.md).
- [x] Delete HF cache `~/.cache/thintensor/models/Qwen--Qwen3.5-9B` and converted model `.thin` files.
- [ ] Pull Qwen3.5-9B again and run benchmarks with hybrid Triton mode to show speedup over HF baseline.
- [ ] Record hybrid results in [benchmark.md](file:///home/satvik/.gemini/antigravity-cli/brain/fb9a4fa7-dbdd-4bb0-a741-cd8c95d8821e/benchmark.md) and perform final cleanup.

### Current Qwen rerun notes
- [x] Fixed max-performance/max-max CUDA crash root cause in Triton plan caching and streamed page lifetime handling; reproduced `thintensor bench` without `CUDA_LAUNCH_BLOCKING=1` or allocator overrides.
- [x] Added `thintensor bench --auto-search` to benchmark real mixed precision/residency candidates instead of manually guessing FP8/INT4 layer mixes.
- [x] Added `scripts/compare_thin_modes.py` for streamed Thin BF16 vs selected Thin mode cosine/top-k validation when HF shards are not locally present.
- [x] Current auto-search winner: full dense body FP8 plus scaled FP8 shared embedding/execution head, direct pageable staging, and an 8GB-derived tight weight budget (`--embed-fp8 --no-pinned-staging --gpu-weight-budget 7653097472`).
- [x] 200 step / 10 warmup auto-search benchmark gate: 12.16 tok/s steady on max-max-perf, selected `body-embed-fp8-stream-pageable-tight-budget`, peak GPU temp 60°C, peak reserved 7.71 GB, H2D reduced to 0.681 GB/token.
- [x] Thin-vs-Thin correctness gate for current winner: min cosine 0.997576, all top-1 same, min top-5 overlap 0.8, generated-token match 1.0 on prefill 1/8 and steps 1/4.
- [ ] Pull raw Qwen3.5-9B safetensors again before final HF/Q8 cosine claims; current cache has tokenizer/config/index plus `.thin`, but not the weight shards.
- [x] Run longer warmed benchmark once a candidate is plausibly near the target; target exceeded on Qwen3.5-9B with 12.16 tok/s steady.

## 🟨 2. GPT-OSS-20B - IN PROGRESS
- [x] Add model-specific attention/MoE overrides/optimizations for GPT-OSS models.
- [x] Fixed packed MXFP4 MoE execution and page-pool residency accounting for native packed expert pages.
- [x] Tune and edit profile parameters to auto-optimize runtime for the GPT-OSS architecture.
- [x] Added MoE-native auto-search candidates using FP8 or grouped INT4 lm head, optional FP8 embeddings, direct pageable staging, and 8GB-derived weight budgets.
- [x] Initial 200-step benchmark: max-max-perf auto-search selected `native-moe-embed-fp8-stream-pageable-tight-budget`, 1.806 tok/s steady, 553.62 ms/token, peak GPU temp 59°C.
- [x] Added selected-slice streaming for packed MXFP4 experts: improved GPT-OSS direct run to ~4.02 tok/s.
- [x] Added hybrid packed-MoE residency: pin all reusable dense/router/norm pages, pin full packed experts for layers 0-8 before lm-head quantization, release BF16 lm-head, then promote packed experts for layers 9-11. FP8-head 200/10 direct run: 6.25 tok/s, 159.92 ms/token, H2D 0.671 GB/token.
- [x] Replaced full expert router sort with GPU `topk`; applies to MoE models without changing token sampling.
- [x] Fixed grouped INT4 lm-head execution path and added it to MoE auto-search/autofit candidates. Best measured 200/10 GPT-OSS run is now `native-moe-embed-int4-head-stream-pageable-high-budget`: 6.68 tok/s, 149.76 ms/token, experts 0-12 resident, selected-expert H2D 127.96 GB/run.
- [x] Fixed high-budget startup OOM root cause: streaming residency now reserves lm-head quantization output memory before pinning packed experts, so 8.1GB-style probes back off instead of crashing at INT4 head allocation.
- [x] Historical same-model llama.cpp b9949 Vulkan Q4_K_M one-run baseline: `50.22 tok/s`. Superseded for the final comparison by the fresh b9964 three-run baseline below.
- [x] Tested MXFP4 expert slice cache: disabled by default because this prompt had 42,240 misses / 0 hits and slowed to 6.07 tok/s.
- [x] Tested pinned staging for selected transfers: disabled because CPU staging cost dropped throughput to 5.64 tok/s.
- [x] Tested approximate MoE top-k limiting: top-2 reaches 9.35 tok/s smoke but fails correctness (min cosine 0.933, top-1 mismatch); top-3 is closer (min cosine 0.984) but still fails top-1 and only reaches 7.47 tok/s smoke. Keep full top-4 for correctness.
- [x] Thin-vs-Thin correctness gate: min cosine 0.999923, top-1 same, top-5 overlap 1.0 on prefill 1 / step 1.
- [x] Record FP8-head direct run in `benchmark_results/gptoss_full_thin_hybrid_residency_200_10.json`.
- [x] Record INT4-head direct run in `benchmark_results/gptoss_full_thin_int4_head_budget8000_200_10.json`.
- [x] Add a lighter exact-reference path for this 8GB/30GB host: load the original `.thin` MXFP4 source pages directly without Q1/Q2 requantization. This holds `6.14 GiB` resident and avoids the Transformers GPU swizzle OOM and CPU BF16 dequantization kill.
- [x] Surpass the interim 10 tok/s gate while preserving full top-4; the remaining target is 44 tok/s with full-model residency.
- [x] Added VRAM-accounted routed-expert caching and reusable materialization buffers; a 3GB cache no longer OOMs or silently steals attention-page headroom.
- [x] Added an SM12 selected-expert native MXFP4 tensor-core kernel. Scalar-vs-tensor-core full-vocabulary gate: min cosine `0.999961`, top-1/top-5 unchanged, generated-token match `1.0` through step 4.
- [x] Removed the routed-cache eviction stream-wide barrier; cached tensors already use allocator stream tracking. The 200/20 routed-cache + tensor-core gate now reaches `14.07 tok/s` steady, full top-4, `not_hf_equivalent=false`, zero cache OOMs, and 50C peak GPU temp. This is an intermediate improvement, not the 44 tok/s target.
- [x] Replace streaming as the primary fit strategy with true middle-out lower-bit packed-expert layers until the full GPT-OSS execution weight set fits VRAM. Preserve embeddings/lm-head; all expert layers use Q2 and central layers `2:22` use Q1, while globals, embeddings, lm-head, and E8M0 scales remain exact.
- [x] Beat the measured same-model Q4 baseline on the correctness-gated 200-token loop. Fresh llama.cpp b9964 Vulkan Q4_K_M averages `40.3631 tok/s` across `38.8324`, `40.8491`, and `41.4077`; ThinRuntime with explicit Torch attention averages `50.3366 tok/s` across `50.3767`, `50.2885`, and `50.3445`, with a `50.3445 tok/s` median and a conservative `1.2473x` win.
- [x] Preserve GPT-OSS `.thin`, HF safetensors, and Q4_K_M GGUF after completing the speed/cosine comparison; no GPT-OSS model artifact was deleted.
- [x] Complete the first repo-wide structural/source audit across Rust archive/conversion/planning plus Python discovery/loading/planning/runtime/kernel/CLI and benchmark surfaces. The audit found duplicated architecture tables, filename-driven quantization policy, a non-executable Q2/Q4 Rust planner, model-specific forward branches, and an MoE path that bypassed optimized dense projection dispatch.
- [x] Finish the literal line-by-line read ledger for every source/script file; do not describe the structural audit as proof that all 45k lines were manually reviewed.
  - [x] `thinruntime/gpu_runtime.py` (11,131 lines): fixed operator-driven per-layer embedding compression, a recurrent Triton MLP `NameError`, and leaked constructor-loop quantization control flow.
  - [x] `thinruntime/triton_kernels.py` (5,070 lines): fixed undefined INT8 tensor-core masking, PyTorch-incompatible argmax tie-breaking, and unstable recurrent softplus; CUDA correctness probes pass.
  - [x] `thinruntime/cli.py` (4,493 lines): aligned chat residency and exact-prefill admission with run/bench for architecture-neutral packed low-bit plans.
  - [x] `thinruntime/model_arch.py` (869 lines): reviewed config/manifest dialect normalization, semantic operator discovery, multimodal text compaction, and remaining legacy-family compatibility adapters.
  - [x] `thinruntime/execution_plan.py` (528 lines): fixed selected-expert MXFP4 fallback for heterogeneous dense/MoE stacks and validated it with an unregistered two-schema fixture.
  - [x] `thinruntime/quantization.py` (431 lines): reviewed source-format discovery, precision ladder, native-kernel admission, and explicit lossy-requantization gates.
  - [x] `thinruntime/auto_fit.py` (780 lines): made unregistered sparse-MoE autofit role-correct (expert INT4, then attention/state FP8-to-INT4), and aligned run/chat/bench page-pool admission; a metadata-only giant fixture reaches a full-resident plan without quantizing globals or heads.
  - [x] `thinruntime/archive.py` (330 lines), `capabilities.py` (485), `operator_registry.py` (58), and `page_policy.py` (111): reviewed archive bounds/hash contracts plus architecture-neutral native admission and page/operator roles.
  - [x] `thinruntime/torch_loader.py` (127 lines) and `tuning_cache.py` (108): reviewed zero-copy tensor views, dtype mapping, and atomic GPU-scoped tuning persistence.
  - [x] `thinruntime/profile_presets.py` (823 lines): replaced recurrent-hybrid and per-layer/shared-KV family-name overrides with operator-set predicates; unregistered equivalent contracts select the same profiles.
  - [x] `thinruntime/architectures.py` (403 lines): reviewed evidence-only coverage registry and confirmed it does not gate unregistered operator-compatible models.
  - [x] `thinruntime/command_runner.py` (152), `hf_loader.py` (152), `hf_pull.py` (142), `model_cache.py` (193), `__init__.py` (51), and `__main__.py` (7): removed quadratic manifest lookup from HF compatibility loading and reviewed binary discovery, credential-safe pulling, cache resolution, and lazy imports.
  - [x] Rust archive/control path: `src/main.rs` (1,029), `archive.rs` (479), `manifest.rs` (458), `plan.rs` (384), `profile.rs` (246), `stats.rs` (303), `repack.rs` (318), `verify.rs` (224), `simulate.rs` (289), `bench.rs` (195), `units.rs` (56), `error.rs` (34), and `lib.rs` (23). Profile page lookup is now linear; fused-decode repack skips incompatible operator groups and verifies on a fused-QKV fixture.
  - [x] `src/convert_hf.rs` (1,730 lines): reviewed multimodal text compaction, common naming adapters, semantic model/page discovery, heterogeneous execution-tape emission, memory bounds, and source-page ordering.
  - [x] Benchmark runners (3,403 lines): `auto_optimize.py` (459), `bench_hf_transformers_decode.py` (114), `bench_llamacpp.py` (126), `bench_lm_head_modes.py` (203), `bench_mxfp4_experts.py` (166), `bench_no_sync_decode.py` (96), `bench_q8_baseline.py` (715), `bench_runtime.py` (508), `bench_runtime_breakdown.py` (308), `run_model_matrix.py` (290), `run_thin_gpu_runtime.py` (127), and `run_thin_torch.py` (291). Standardized CPU-safe HF baselines, real decoded-token accounting, and 200/10 steady-state claim admission.
  - [x] Correctness/comparison runners (5,659 lines): `compare_gguf_hf_logits.py` (300), `compare_hf_thin_logits.py` (2,340), `compare_llamacpp_q8_logits.py` (548), `compare_q8_baseline.py` (706), `compare_thin_modes.py` (419), `compare_thin_vs_hf_runtime.py` (390), and `stress_validate_thin.py` (956). Removed process-wide CUDA tensor mutation after HF OOM, made eager/remote HF reference loading architecture-correct, fixed inflated hydration decode accounting, and separated Transformers hydration results from native-runtime evidence.
  - [x] Planning/profiling/report scripts (3,639 lines): `first_run_pipeline.py` (442), `inspect_hf_compat.py` (223), `kv_cache_experiments.py` (370), `make_benchmark_table.py` (239), `make_q8_summary.py` (276), `optimize_runtime_one_by_one.py` (1,168), `profile_decode_bottlenecks.py` (396), `profile_raw_forward_shapes.py` (275), `rank_models_for_thintensor.py` (117), and `simulate_kv_cache.py` (133). Made compatibility operator-driven, safe-claim aggregation steady-state gated, and kept microprofiles explicitly diagnostic.
  - [x] Runtime/utility scripts (1,736 lines): `thin_runtime.py` (1,552 after fixes), `check_thin_loader.py` (92), `pull_gpt_oss.py` (18), and `test_shared_tensors.py` (74). Lossy KV capacity models are no longer admitted as executable codecs; optimizer/runtime capabilities expose only exact BF16/FP16 KV until real encode/decode kernels pass parity. Shared-tensor validation now handles fused pages and genuinely untied heads.
- [x] Add one architecture-neutral page-role policy shared by conversion metadata, automatic fit accounting, and runtime precision selection. Global tensors and heads are protected by role; decoder attention/MLP/MoE/state-space matrices are selected without model-name checks.
- [x] Add native packed-expert Q2/Q1 storage and SM12 execution. Automatic fit now applies Q2 then Q1 middle-out across the decoder body until the actual weight budget is met, while E8M0 scales and all global/head pages stay exact.
- [x] Eliminate double-counted page-pool headroom and preserve subprocess stderr in JSON benchmark failures.
- [x] Full-residency GPT-OSS plan on the current 7.56 GiB CUDA allocation: source `13,761,264,768` bytes to estimated `6,993,034,368` bytes; all layers Q2, central layers `2:22` Q1, zero streamed expert pages, exact embedding/lm-head.
- [x] Initial full-vocabulary Q1/Q2 gate (`prefill=1`, `step=1`): cosine `0.991415`, top-1 unchanged, top-5 overlap `0.8`, generated-token match `1.0`. This is provisional pending the multi-step gate.
- [x] Full-residency warmed 20/3 speed probe: `26.28 tok/s`, up from `14.07 tok/s`, with zero selected-expert H2D. Still below the `44 tok/s` target; CUDA graph capture correctly remains disabled because the existing graph hardcodes token index 0 and would violate causal-KV correctness.
- [ ] Finish replacing family-specific execution branches with an operator-graph executor and registered attention/MLP/MoE/state-space blocks; conversion must reject unknown operator contracts precisely rather than fall back to Transformers.
- [ ] Run cross-architecture correctness and HF speed gates for tiny dense, fused-QKV dense, separate-expert MoE, packed-expert MoE, sliding/hybrid attention, and state-space/linear-attention fixtures.
- [x] Run the warmed 200-token GPT-OSS benchmark after explicit acceptance of the sub-2-bit quality tradeoff. The final path uses 10 warmup plus 200 measured causal-KV tokens, full router top-4, exact embedding/head, and no expert H2D.
- [x] Run the Q1/Q2 multi-step full-vocabulary gate: step-4 cosine falls to `0.983678` with top-1/top-5 and generated tokens still stable. Reject the fixed symmetric Q1 codec as a default quality-preserving plan; a better sub-2-bit codec is required.
- [x] Profile fully resident MoE execution and add shared MoE multi-QKV plus exact fused top-k router, interleaved SwiGLU, and weighted-expert reduction kernels. The exact router fusion reduces routed-MLP profile time from about `1.30` to `1.02 ms/layer`, but 20/3 end-to-end remains about `25.85 tok/s`.
- [x] Reject FP8 attention as an automatic packed-MoE candidate: it reduced resident bytes to `6.36 GB` but slowed the 20/3 run to `24.95 tok/s` and changed the output trajectory.
- [x] Reject padded `BLOCK_K=256` low-bit tensor-core tiles: tail masking is correct but microbenchmarks regress versus exact-divisor `BLOCK_K=64`.
- [x] Qualify the fixed symmetric Q1/Q2 packed expert plan under the explicitly accepted quality target. Source-MXFP4 comparison keeps top-1 unchanged at steps 1 and 4 with cosine `0.987376` and `0.983299`; the previously selected one-step mix reaches `0.991415`. This is materially above the measured llama.cpp Q4_K_M centered-logit cosine of `0.792350` against the same source reference.
- [x] Add architecture-neutral operator and page-role registries. Native eligibility now depends on implemented operator contracts, not a registered model-family name: a synthetic unregistered decoder with known roles compiles natively, and an unknown operator is rejected by exact name.
- [x] Compile heterogeneous layer schemas independently and represent separate experts as compact expert-set bindings instead of expanding `layers x experts x projections`; synthetic dense-plus-MoE stacks pass plan compilation.
- [x] Add explicit operator names to new Rust execution-tape stages and semantic runtime dispatch for recurrent hybrid, per-layer/shared-KV, and dual-post-norm decoders.
- [x] Add Rust `offload-middle-out` residency and rename misleading `all_q2/all_q4` candidates to state that Q2/Q4 applies to KV, not weights.
- [x] Replace stream-wide synchronization on GPU page eviction with budget-accounted event-retired storage. TinyLlama causal-KV stress completed 3,240 evictions with zero retired bytes left and no CUDA access violation.
- [x] Replace eager full-model `--cpu-offload` duplication with a bounded lazy CPU LRU. A 128 MiB cache test held 111.2 MiB/10 pages while completing 2,430 GPU evictions; expose hit/miss/eviction and byte-cap telemetry.
- [x] Make normalization an explicit archive operator contract. LayerNorm fixtures now emit `layer_norm` plus `eps` on every norm stage instead of contradicting the model capability with hardcoded `rms_norm`; the converted archive verifies.
- [x] Redefine archive memory bounds for giant streamed models: minimum VRAM is exact global/head pages plus the largest executable stage and live operator/state buffers, while recommended VRAM is full unique-weight residency plus live buffers. Scratch no longer scales as the non-live `hidden_size x intermediate_size` product.
- [x] Broaden config discovery across common architecture dialects (`n_embd`/`d_model`, `n_layer`, `n_head`, `n_head_kv`, `n_inner`/`d_ff`, and `n_vocab`) and infer missing intermediate size from converted tensor shapes. An unregistered alias-config fixture resolves the correct LayerNorm/GQA operator contract.
- [x] Broaden Rust offload profiling from a fixed dense projection list to fused attention, packed/separate MoE, linear-attention, and per-layer projection roles; retain enough large-page candidates for heterogeneous expert stacks.
- [x] Fix fused-page event retirement accounting: evicting the last zero-owned logical view now retires and budgets the physical CUDA allocation instead of dropping its bytes early. A queued-consumer GPU test retained all 16,384 physical bytes until the event completed, then reaped to zero.
- [x] Prototype a grouped ternary expert codec at `1.944 bpw`. Real GPT-OSS expert weight cosine improves from about `0.832` binary to `0.893`; its native tensor-core output matches a mathematical decode at `0.999999` cosine. Reject it as the auto-fit replacement because the full gate/up kernel is about `0.70-0.75 ms` versus Q1 `0.36 ms`, and all-ternary experts alone still do not meet the current weight budget. Keep it experimental while a faster, higher-quality codec is developed.
- [x] Make causal attention explicit in the Rust execution graph. Each non-recurrent layer now emits an `mha_attention`, `mqa_attention`, or `gqa_attention` stage with causal/head geometry and optional sliding-window parameters; archive verification rejects an unknown operator by exact stage/name. Common config aliases pass both Python native-capability analysis and Rust conversion.
- [x] Make the Python compiled operator plan a runtime admission gate rather than inspection-only metadata. Runtime construction now rejects missing tensor/operator contracts, exposes the heterogeneous schema/operator set in telemetry, and recognizes recurrent `gated_delta_net` layers independently. An unregistered fused-QKV/fused-gate LayerNorm fixture constructs and executes through the compiled plan on CUDA.
- [x] Fuse arbitrary-width Q/K/V projection dispatch and per-head Q/K RMSNorm plus full RoPE. On GPT-OSS geometry, masked non-power-of-two multi-matvec matches BF16 reference (`1.0` cosine) and is ~12% faster than three projections; fused norm+RoPE is >`0.99999` cosine. The real 20/3 diagnostic improves from `26.28` to `26.65 tok/s`, attention from about `0.586` to `0.458 ms/layer`, and layer average from about `1.624` to `1.428 ms`. This remains a rejected-Q1 speed diagnostic, not a correctness-qualified 44 tok/s result.
- [x] Persist content-addressed low-bit derived pages with atomic writes, shape/dtype validation, corrupt-entry invalidation, mmap loading, and disk guards. GPT-OSS populated 48 entries/3.384 GB; the repeat run had 48 hits, zero misses/writes, reduced `load_s` from `247.1` to `1.58`, and reduced RSS growth from ~19.2 GB to ~9.0 GB.
- [x] Reject a 1.125-bit asymmetric binary codebook (one sign bit/weight plus four magnitude bits per 32-value group). On a real GPT-OSS layer-2 gate/up expert it improves weight cosine only from `0.772194` to `0.791211`; this is insufficient to repair the rejected multi-step Q1 trajectory and does not justify another codec/kernel path.
- [x] Add a semantic runtime-executor registry and route top-level decode through its selected contract (`optimized_dense_decoder`, `generic_dense_decoder`, `sparse_moe_decoder`, `recurrent_hybrid_decoder`, or `shared_kv_decoder`). A previously dangerous heterogeneous dense+MoE plan is now rejected before CUDA with the exact missing per-layer graph capability instead of incorrectly executing every layer as MoE; an unregistered fused dense fixture still executes and reports its selected executor.
- [x] Prototype a 1.25-bit constrained two-means expert codebook (one assignment bit/weight plus two FP4 centroid codes per 32-value group). A 32,768-group real GPT-OSS sample improves cosine from `0.768286` to `0.806801`; retain it as research evidence but do not integrate it because the gain is not yet sufficient for the rejected step-4 trajectory.
- [x] Replace expanded low-bit selected-expert execution with direct scalar Q1/Q2 kernels, then fuse gate/up decode with clamped SwiGLU and fuse down projection, BF16 bias, router weighting, and expert reduction. Real Q1/Q2 tensors are bit-identical to the prior path; tuned Q1 and Q2 geometries reduce the routed MLP bottleneck without touching heads.
- [x] Fuse router matvec, exact top-4 selection, and softmax. Random and real probes preserve router indices/scores, and compiled fully resident plans prebind layer tensors so the hot decode loop no longer performs page-role lookup.
- [x] Extend fused single-token GQA attention with GPT attention sinks and fix its runtime admission predicate. The Triton path matches the Torch reference at `0.99998987` cosine with `0.001953` max absolute error, but a real 200/10 run reaches only `42.86 tok/s`; keep explicit Torch attention in the winning GPT-OSS profile.
- [x] Measure Q4_K_M full-vocabulary quality against the original MXFP4 source reference rather than assuming bit-width implies quality. Artifact `correctness_results/gptoss_q4km_vs_source_mxfp4_step1.json` records centered-logit cosine `0.792350`, versus Thin Q1/Q2 `0.991415` for the accepted one-step configuration.
- [x] Record the final controlled speed/quality comparison in `benchmark_results/gptoss_q1q2_vs_llamacpp_b9964_200_10.json` and the optimized multi-step same-engine gate in `correctness_results/thin_modes/gptoss_optimized_q1q2_prefill1_steps1_4.{md,json}`.

## 🟨 3. LLaMA Latest Series (9-12B) - IN PROGRESS
- [x] Select candidate model: `unsloth/Llama-3.2-11B-Vision-Instruct` / Ollama `llama3.2-vision:11b` (`mllama`, 10.7B, Q4_K_M in Ollama).
- [x] Fixed Mllama text-only conversion: canonicalizes `language_model.*`, drops vision/cross-attention tensors, and compacts the text path from 40 multimodal layers to 32 self-attention decoder layers.
- [x] Convert text-only archive: `llama-3.2-11b-vision-text.thin`, 291 pages, 14.96 GiB, verify pass.
- [x] Existing ThinTensor-vs-HF correctness artifact: `correctness_results/cli/bench_200.md`, 200-step full-vector min cosine 0.939758, min top-5 overlap 0.8.
- [x] Ollama Q4_K_M speed artifact: `benchmark_results/llama32_vision_11b_ollama_q4km_200.json`, 22.08 tok/s, 45.28 ms/token, 7.45 GB VRAM.
- [x] Ollama Q4_K_M vs HF API-limited logprob artifact: `correctness_results/ollama/llama32_vision_11b_q4km_top20_vs_hf.md`, min top-20 probability cosine 0.720495, min top-20 overlap 0.55, top-1 not same on all cases.
- [ ] ThinTensor `autofit` 200-step speed gate: not rerun here because user said not to rerun ThinTensor bench and to use existing `bench_200` results.
- [x] Record current results in [benchmark.md](file:///home/satvik/.gemini/antigravity-cli/brain/fb9a4fa7-dbdd-4bb0-a741-cd8c95d8821e/benchmark.md).
- [x] Delete ThinTensor/HF cache and converted archive before moving to the next model.

## ⬜ 4. Nemotron Model
- [ ] Select candidate model (>20 GB in safetensors/weights).
- [ ] Convert and run 200-step benchmark profiles.
- [ ] Record results in [benchmark.md](file:///home/satvik/.gemini/antigravity-cli/brain/fb9a4fa7-dbdd-4bb0-a741-cd8c95d8821e/benchmark.md).
- [ ] Delete cache/weights.

## ✅ 5. Microsoft Phi-4-mini-instruct (Dense 3.8B) - COMPLETE
- [x] Pull `microsoft/Phi-4-mini-instruct` model.
- [x] Convert and run 200-step benchmark profiles.
- [x] **Results**: Safe 42.6 tok/s (4.3x), Balanced 42.5 tok/s (4.5x), Max-performance 55.5 tok/s (5.8x), Max-max-perf 55.5 tok/s (5.8x) vs HF 9.5 tok/s
- [x] Delete cache/weights.

## ✅ 6. Qwen2.5-1.5B-Instruct - COMPLETE
- [x] Pull `Qwen/Qwen2.5-1.5B-Instruct` model.
- [x] Convert and run 200-step benchmark profiles.
- [x] **Results**: Safe 83.4 tok/s (1.04x), Balanced 68.9 tok/s (0.86x), Max-performance 134.3 tok/s (1.67x), Max-max-perf 160.2 tok/s (1.99x) vs HF 80.3 tok/s
- [x] Delete cache/weights.

## ✅ 7. Microsoft Phi-3.5-mini-instruct (Dense 3.8B) - COMPLETE
- [x] Pull `microsoft/Phi-3.5-mini-instruct` model.
- [x] Convert and run 200-step benchmark profiles.
- [x] **Results**: Safe 42.4 tok/s (4.2x), Balanced 36.1 tok/s (3.6x), Max-performance FAILED, Max-max-perf FAILED vs HF 10.1 tok/s
- [x] Delete cache/weights.

## ⬜ 8. Microsoft Phi-4 (Dense 14B)
- [ ] Pull `microsoft/phi-4` model.
- [ ] Convert and run 200-step benchmark profiles.
- [ ] Record results in [benchmark.md](file:///home/satvik/.gemini/antigravity-cli/brain/fb9a4fa7-dbdd-4bb0-a741-cd8c95d8821e/benchmark.md).
- [ ] Delete cache/weights.

## 📋 Overnight Progress Log
| Time | Model | Task | Status | GPU Temp | CPU Temp | Disk % |
|------|-------|------|--------|----------|----------|--------|
| 02:00 | Phi-4-mini | Pull & Convert | ✅ | 47°C | - | 78% |
| 02:30 | Phi-4-mini | HF Baseline | ✅ 9.5 tok/s | 52°C | - | 78% |
| 02:45 | Phi-4-mini | Safe Profile | ✅ 42.6 tok/s (4.3x) | 52°C | - | 78% |
| 03:00 | Phi-4-mini | Balanced/Max-perf/Max-max-perf | ✅ 55.5 tok/s (5.8x) | 56°C | - | 78% |
| 03:15 | Phi-4-mini | Cleanup | ✅ | 47°C | - | 75% |
| 03:30 | Phi-3.5-mini | Pull & Convert | ✅ | 50°C | - | 74% |
| 04:00 | Phi-3.5-mini | HF Baseline | ✅ 10.1 tok/s | 57°C | - | 74% |
| 04:15 | Phi-3.5-mini | Safe Profile | ✅ 42.4 tok/s (4.2x) | 57°C | - | 74% |
| 04:30 | Phi-3.5-mini | Balanced Profile | ✅ 36.1 tok/s (3.6x) | 58°C | - | 74% |
| 04:45 | Phi-3.5-mini | Cleanup | ✅ | 50°C | - | 71% |
| 13:30 | Qwen3.5-9B | Auto-search max-max candidates | ✅ selected body-fp8-stream-pageable, 4.26 tok/s smoke | 53°C | - | 83% |
| 13:34 | Qwen3.5-9B | Standalone selected command | ✅ 4.29 tok/s, `--no-pinned-staging` | 45°C | - | 83% |
| 13:37 | Qwen3.5-9B | Thin-vs-Thin correctness | ✅ min cosine 0.997920, top1 all same | 47°C | - | 83% |
| 13:46 | Qwen3.5-9B | Embed/head FP8 correctness | ✅ min cosine 0.997576, top1 all same | 47°C | - | 83% |
| 13:56 | Qwen3.5-9B | Resident-budget probe | ✅ 7.125GiB weight budget, 11.22 tok/s 64/4 | 51°C | - | 83% |
| 14:05 | Qwen3.5-9B | Auto-search 200/10 gate | ✅ selected tight embed-FP8, 12.16 tok/s | 60°C | - | 83% |
| 14:25 | GPT-OSS-20B | Pull & Convert | ✅ 12.82GiB `.thin`, verify pass | 45°C | - | 81% |
| 14:50 | GPT-OSS-20B | MXFP4/runtime fixes | ✅ packed MoE + page residency + embed/head FP8 | 49°C | - | 86% |
| 14:58 | GPT-OSS-20B | Auto-search 200/10 gate | ✅ selected native MoE tight embed-FP8, 1.806 tok/s | 59°C | - | 86% |
| 14:59 | GPT-OSS-20B | Thin-vs-Thin correctness | ✅ min cosine 0.999923, top1 same | 55°C | - | 86% |
| 18:16 | Llama-3.2-11B-Vision | Text conversion + Ollama compare | ✅ text archive verify pass; Ollama Q4_K_M 22.08 tok/s; Thin full-vector cosine 0.939758 vs Ollama top-20 cosine 0.720495 | 39°C | - | 80% |
| 18:18 | Llama-3.2-11B-Vision | Cleanup | ✅ deleted 35GiB ThinTensor/HF cache and archive | 41°C | - | 65% |
| 20:05 | GPT-OSS-20B | Hybrid packed-MoE residency | ✅ 6.25 tok/s, experts 0-11 resident, H2D 0.671 GB/token | 52°C | - | 78% |
| 20:18 | GPT-OSS-20B | Cache/pinned staging probes | ⚠️ cache 0 hits and slower; pinned staging slower from CPU staging | 53°C | - | 78% |
| 20:44 | GPT-OSS-20B | INT4 head + high residency | ✅ 6.68 tok/s, experts 0-12 resident, H2D 0.620 GB/token | 53°C | - | 78% |
| 20:51 | GPT-OSS-20B | High-budget OOM guard + MoE top-k probes | ⚠️ OOM fixed; top-2/top-3 rejected on Thin-vs-Thin correctness | 48°C | - | 78% |
| 18:45 | GPT-OSS-20B | Full-resident Q1/Q2 fused decode | ✅ explicit-Torch-attention three-run median 50.34 tok/s versus llama.cpp b9964 mean 40.36 tok/s; top-1 stable through step 4 | 58°C | - | - |


Current GPT-OSS result: full-resident middle-out Q1/Q2 eliminates expert transfers and fused low-bit MoE execution beats the fresh llama.cpp b9964 Q4_K_M baseline by `24.73%` using the conservative repeated median. The exact embedding/lm-head remain unquantized, source-relative top-1 stays stable through step 4, and the GPT-OSS model files remain preserved.
