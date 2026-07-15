# Qwen3.6-27B optimization log

## 2026-07-14 target resolution and disk gate

- Hypothesis: the requested model and a same-variant Q4_K_M now exist, but the required staged workflow is necessary to preserve 12 GiB of free space.
- Repository: `main` at `908fc2a8107366d24beb7b040ed9f21e661c5189`; dirty-tree fingerprint `7d6f376fc612e6e14927ed2c2c10c43aca6a3915d0f5870214c2920df28c758c` before new Qwen27B artifacts.
- Target: official public ungated `Qwen/Qwen3.6-27B` revision `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`, 15 BF16 shards and 55,563,006,400 download bytes.
- Model identity: current weight LFS hashes exactly match first-upload revision `1b559cf7215ebe67ff10758e14f6293ba883223b`; later official revisions only changed metadata, tokenizer/generation configuration, license, and README.
- Q4 baseline: `unsloth/Qwen3.6-27B-GGUF` revision `82d411acf4a06cfb8d9b073a5211bf410bfc29bf`, file `Qwen3.6-27B-Q4_K_M.gguf`, 16,817,244,384 bytes, declared base `Qwen/Qwen3.6-27B`.
- Architecture: 64 text layers, 48 gated-delta-net layers, 16 full-attention layers, 5,120 hidden size, 248,320 vocabulary, BF16 source.
- Disk before download: 113,250,140,160 bytes available on `/`; repository uses 1,004,521,579 bytes.
- Disk calculation for Phase A: 113,250,140,160 - 16,817,244,384 Q4 - 31,206,236 llama.cpp archive - 12 GiB reserve = 83,516,788,276 bytes positive.
- Decision: accept Phase A. Download only the Q4_K_M and the verified official b9986 Vulkan binary, then benchmark/dump logits before deleting the GGUF.
- Conversion audit: the current converter mmaps all shards and writes one final archive but has no `--streaming-pack`, `--consume-source-shards`, minimum-free-space, journal, or resume support. Phase B cannot begin safely until this critical path is implemented and tested on a small sharded fixture.

## 2026-07-14 transactional conversion implementation

- Hypothesis: a deterministic archive prefix plus a durable source-range plan is enough to resume after source shards have already been consumed.
- Implementation: added a resumable `.thin.partial` writer, atomic plan and journal sidecars, per-page free-space gates, per-shard fsync, page-range read-back BLAKE3 verification, direct-HF-directory deletion guards, and source consumption only after the last dependent page verifies.
- Forced-crash test: converted a synthetic 258-MiB two-shard Llama fixture with source consumption, observed shard 1 recorded as verified/deleted after 11 pages, then sent SIGKILL (status 137). The resume began with shard 1 absent, truncated the uncommitted tail to the journal boundary, wrote page 12 from shard 2, consumed shard 2, and produced `verify: ok` with `completed=true`.
- Additional fix: the test exposed that `build_memory_plan` counted global embedding/head pages once in `global_bytes` and again as the largest execution stage. The largest stage is now restricted to layer-owned tensors; the test archive then validated.
- Tests: `cargo check --all-targets`, `cargo test --all-targets`, Python compileall, `git diff --check`, forced SIGKILL/resume conversion, and post-resume core archive verification.
- HF reference path: added a bounded-memory reference that uses the official `Qwen3_5DecoderLayer` implementation, maps one SafeTensors layer at a time, slices embedding rows, and chunks the vocabulary head. On a synthetic Qwen3.5 fixture its step-1 and step-3 logits were bit-identical to a fully resident `Qwen3_5TextModel` (`max_abs_error=0`, `mean_abs_error=0`).
- Decision: accept both capabilities. The synthetic artifacts are disposable and are removed after retaining this evidence.

## 2026-07-14 Phase A Q4_K_M baseline

- Download verification: exact 16,817,244,384-byte `Qwen3.6-27B-Q4_K_M.gguf`; SHA-256 `5ed60d0af4650a854b1755bd392f9aef4872643dc25a254bc68043fa638392a0` matches pinned Hugging Face LFS metadata.
- llama.cpp: official b9986 Vulkan archive SHA-256 `6ec4c41dbb17590cf0dfbb21ccb842b8e235dc72d9e9556c1db9fe3bc390e768`, commit `91c631b21d6e5d09e9c6659efdf6baeef5a44ddb`; device `Vulkan1` is the RTX 5050 Laptop GPU.
- Identity at load: `qwen35 27B Q4_K - Medium`, 26,895,998,464 parameters and 16,806,250,496 tensor bytes.
- Sustained single-sequence decode: 4.579955 tok/s mean over 200 generated tokens and three repetitions, samples `4.57234 / 4.58493 / 4.58260`, batch/ubatch 1, F16 KV, 512-MiB fit margin and 2,048-token minimum fit context.
- Telemetry: maximum sampled RTX temperature 58C, power 21.04 W, and memory 7,006 MiB.
- Full-vocabulary evidence: saved all 248,320 pre-sampling log-probabilities at fixed-teacher steps 1/8/32; every value is finite and NPZ SHA-256 is `30383687da26ccd19207ec2295c963fd11463278c5e93bb417586bb8cf607e9a`. Prompt length is 12 tokens and the fixed continuation is 44 tokens.
- Artifacts: `benchmark_results/qwen27b/llamacpp_q4km_b9986_200_r3.json`, its raw JSON/log/thermal CSV, and `correctness_results/qwen27b/q4km_b9986_fixed_teacher.{json,npz}`.
- Decision: baseline and correctness evidence are complete and readable. Mark the Q4 file safe to delete before Phase B.

## 2026-07-14 conversion planning and critical-path audit

- Dry-run gap: added `thintensor convert --dry-run` through both public Python and Rust CLIs. It computes the exact header/manifest/page-table/payload archive size, deterministic shard order, each shard's last dependent page, upper-bounded plan/journal temporary bytes, and a page-by-page minimum-free-space simulation with source consumption.
- Dry-run validation: a 12-page synthetic archive reported exactly 9,690 output bytes and the final deletion point at `lm_head.weight`; the command wrote no archive, partial, plan, or journal and retained its source shard.
- Archive validation fix: unreferenced physical page-table entries now fail instead of being silently accepted. Physical fused pages remain accepted only when a logical manifest page names them through `fused_to`.
- Architecture validation fix: malformed `cross_attention_layers` values now produce an explicit error in both capability discovery and runtime architecture discovery instead of disappearing from the layer map.
- Profiler fix: unavailable/unrecorded CUDA event pairs now yield `null` fields plus `timing_unavailable` evidence; they can no longer silently reduce a component timing toward zero.
- Memory ownership: added `thinruntime/memory_report.py`, split page weights, registered external head, async-retired tensors, expert cache, quantization sidecars, scratch, KV, CPU/pinned cache, CUDA allocation/reserve, and mmap virtual bytes. Fixed a real double count where a page-pool-registered quantized head was also added again by `resident_weight_bytes`.
- Tests: synthetic dry-run non-mutation, orphan/fused-page validation, malformed/valid cross-attention IDs, component accounting, LM-head overlap, `cargo check --all-targets`, `cargo test --all-targets`, Python compileall, and `git diff --check`.

## 2026-07-14 dense Q2/E8M0 resident mechanism

- Hypothesis: dense MXFP4 cannot fit the 27B body in 8 GB, but Q2 groups plus one-byte E8M0 scales per 32 weights can reduce the text body near 6.1 GB and reuse the existing Blackwell scaled tensor-core kernel.
- Implementation: added MSE-selected direct BF16-to-Q1/Q2 packing, lazy page-pool planning and quantization, a one-expert tensor-core dispatch for dense matrices, full-resident CPU-cache release, checksum-keyed derived caches, runtime flags, telemetry, and an experimental public auto-search candidate with exact CPU embedding and grouped-128 INT4 head.
- Kernel validation: on a deterministic 128x128 BF16 matrix, Q2 reconstruction MSE was `0.0003138632`; the RTX 5050 tensor-core result versus explicit dequantized matvec had maximum absolute error `0.00384545` and cosine `0.99999857`.
- Runtime validation: a converted one-layer causal Llama fixture completed the page-pool route at 1,603.04 tok/s, retained all four causal KV tokens, produced finite logits, and reported 48,384 resident weight bytes. This is a wiring test, not a target-model speed claim.
- Cache validation: first isolated run wrote all seven derived projection caches (66,347 bytes); a second isolated run hit all seven and wrote none.
- Decision: retain behind explicit Q2 flags and the experimental auto-search candidate. Acceptance for Qwen3.6-27B remains gated on the real 200-token speed run and full-vocabulary quality evidence.

## 2026-07-14 multimodal text-schema compatibility correction

- Finding: the pre-conversion inspector compiled raw repository names such as `model.language_model.layers.0.*`, while the converter intentionally writes canonical `model.layers.0.*` text names. On Qwen3.6 this produced false missing-role errors for all 64 layers even though the archive would have been canonicalized.
- Implementation: the inspector now applies the converter-equivalent multimodal text projection, retains the root untied `lm_head.weight`, removes vision/adapter pages, compacts declared cross-attention layers, and reports source/canonical/excluded tensor counts separately. Native readiness is now also conditional on a successful schema plan.
- Official-index validation: 1,199 repository tensors project to 851 executable text tensors with 348 non-text tensors excluded. The result compiles as `heterogeneous:gated_delta_net_dense,separate_qkv_gated_dense`, with no schema or native-executor gaps.
- Decision: accept. This prevents a misleading readiness result and validates the exact tensor graph before any destructive source consumption.

## 2026-07-14 archive-backed full-model smoke

- Post-conversion compatibility: all 851 archive tensors compile into the 64-layer heterogeneous recurrent/full-attention graph; the native executor reports no gaps.
- Failure retained: a 7.8-billion-byte BF16 weight allowance OOMed because it left no CUDA activation/page headroom. Reducing it to 5.5 billion bytes then exposed that the protected exact embedding, head, and active MLP set do not coexist at that cap.
- Bug found and fixed: the specialized Qwen3.5 forward indexed the metadata-only tensor used by `--cpu-embed` unless an embedding scale was present. It now dispatches all CPU-mapped embeddings through the exact BF16 row copier. PyTorch exposed the underlying `meta`/CUDA mismatch that Triton had surfaced as an illegal memory access.
- Full smoke: exact BF16 body/head, exact CPU-mapped embedding, 64 layers, explicit 12-token prompt, causal BF16 KV, and one decoded token completed. The generated token was `248068`; KV actual sequence length and tokens attended were both 13.
- Fit/telemetry: 5,608 MiB sampled GPU peak, 5,188,616,704 allocated-byte peak, 48C, 17.61 W, 85% utilization, and no thermal stop. No batching or speculative decoding was enabled.
- Reference cost: the decoded token took 24.318 seconds (`0.0411218 tok/s`) and the run transferred 1,089,061,807,872 bytes across prefill and decode. This is accepted as the correctness smoke, not a performance candidate.

## 2026-07-14 explicit-prompt benchmark provenance

- Finding: the runtime benchmark performed genuine autoregressive causal decode but did not retain its generated token trajectory and the public benchmark could not inject an engine-independent tokenized prompt.
- Implementation: added explicit comma-separated prompt-token prefill to the runtime and public benchmark/auto-search paths, a timed device-side generated-token buffer with one post-timing host copy, prefill timing, initial decode-token disclosure, and prompt-aware token positions. The ten-step post-run profiler now continues at non-overlapping positions and advances its token instead of repeatedly profiling one ID.
- Validation: a converted one-layer causal fixture prefilling `[1,2,3]` generated and retained `[100,12,30]`; its exact BF16 KV telemetry reported actual sequence length and tokens attended of 6 after three decode steps. Public `thintensor bench --dry-run` forwarded the same explicit IDs.
- Decision: accept. The tiny device-side ID copy is included in decode time, making the measurement slightly conservative rather than hiding provenance overhead.

## 2026-07-14 benchmark telemetry and comparison completeness

- Telemetry: extended the existing half-second NVIDIA thermal sampler to capture peak power draw, device-reported memory use, GPU utilization, and sample count in direct and public benchmark JSON. A live idle RTX 5050 sample parsed all fields; target peaks remain pending the sustained run.
- Comparison: full-vocabulary reports now include centered-cosine minimum, mean, and p05/p50/p95; centered maximum/mean absolute errors; top-1/checkpoint-greedy agreement and first divergence; top-5 mean/min overlap and exact order; JS and total variation. Per-candidate NPZ prefixes allow HF, Q4 log-probabilities, and ThinTensor vectors in one report.
- Memory label correction: page-pool external allocations can include both a separate execution head and expert materialization buffers, so the ownership component is now named `external_registered_allocations` instead of incorrectly attributing every byte to the LM head.
- Decision: accept the additive evidence fields. No target speed or quality conclusion is recorded until the exact checkpoint finishes verification, conversion, and real decode.

## 2026-07-14 exact checkpoint reference and consuming conversion

- Checkpoint verification: all 15 pinned SHA-256 hashes, 1,199 index/header tensor names, shard mappings, dtypes, shapes, non-overlapping data ranges, 55,563,006,400 file bytes, and 55,562,855,904 tensor bytes passed.
- Official BF16 reference: the bounded official `Qwen3_5DecoderLayer` path finished 64 layers in 28.42 seconds. All three 248,320-value vectors are finite; NPZ SHA-256 is `6abfc712998f4ae0dba76da32002880d688239dd12e02a8f20ceb504fd89144e`.
- Q4 quality bar versus BF16: minimum/mean centered cosine `0.9875233769 / 0.9910152356`, p05/p50/p95 `0.9876149714 / 0.9884393215 / 0.9962186396`, top-1 agreement `1.0`, minimum/mean top-5 overlap `0.8 / 0.9333333333`, and exact top-5 order rate `0.3333333333`.
- Conversion: the installed public CLI committed and verified 851 text pages, excluded 348 vision/MTP pages, fsynced/read back every source-dependent range, deleted all 15 verified shards, and completed its journal in 137.84 seconds. The archive is 53,792,526,418 bytes and both core and public verification passed; free space rose to 58,959,220,736 bytes.
- Dry-run correction: the preflight forecast was one byte low because placeholder checksum bytes changed decimal JSON widths. This did not affect the 43.03-GB minimum-space safety decision. The code now hashes real payloads in an exact dry-run; the original report and explicit one-byte reconciliation are both retained rather than silently rewriting evidence.

## 2026-07-14 archive-backed calibration and low-precision quality search

- Archive reference equivalence: rerunning the official Transformers decoder one layer at a time from the verified `.thin` archive produced byte-identical step-1/8/32 vectors and the same NPZ SHA-256 `6abfc712998f4ae0dba76da32002880d688239dd12e02a8f20ceb504fd89144e`. The pass also captured 496 per-projection RMS vectors and 32-channel covariance blocks without retaining raw activations.
- Rejected formats: all-body Q2/E8M0, MXFP4, symmetric CPU INT4 g128/g32, affine CPU INT4 g32 without calibration, arbitrary BF16 edge layers, FP8 full-attention layers, and sparse residual corrections all failed the Q4 centered-cosine gate. Q2 down-projection alone collapsed the fixed-teacher logits (minimum raw cosine `-0.162783`), so Q2 is not a quality-safe speed route for this target.
- Quality mechanism: added diagonal-Hessian affine fitting followed by two-pass 32-channel GPTQ error feedback. The packed payload remains standard group-32 INT4 and runs through PyTorch's AVX-512 CPU kernel; calibration changes code selection only. Derived cache keys bind source checksum, group, affine mode, calibration hashes, damping, and quantizer version.
- Head isolation: replacing the grouped-128 INT4 vocabulary head with per-row scaled FP8 raised canonical centered cosine above Q4. ThinTensor minimum/mean are `0.9910904169 / 0.9942892392`, versus Q4_K_M `0.9875233769 / 0.9910152356`; minimum/mean top-5 overlap are tied at `0.8 / 0.9333333333`, and ThinTensor top-1 matches HF at all three checkpoints.
- GPU affine route: added affine zero-point support to the bandwidth-oriented Triton grouped INT4 matvec plus projection-level residency selectors. A real 17,408x5,120 page ran at `0.1756 ms` (`317.2 GB/s`) with cosine `0.9999854` versus the accepted CPU result. Moving up, recurrent-QKV, and full-Q pages preserved quality and fit at 6.56 GiB planned weights.

## 2026-07-14 sustained speed results and blocker

- Quality-accepted CPU profile: public 200-token, 10-warmup, batch-1 causal decode reached `2.117566 tok/s` and `31.71 GB/s` effective end-to-end bandwidth. The CPU INT4 kernels consumed `82.61 / 93.98 s`, reading 3.531 TB across 115,072 projection calls. GPU peak was 6,506 MiB during setup; steady owned page residency was much lower and telemetry records both allocator and ownership views.
- Shared-input transfer grouping cut D2H synchronization batches but only improved a 20-token tuning run from `2.1176` to `2.1709 tok/s`; retained as an exact orchestration cleanup, not claimed as a target win.
- Fully resident affine GPU split with the new fast kernel reached `2.879337 tok/s` in the 20-token diagnostic, peak device memory 7,150 MiB, and no per-token weight thrash. It remains below Q4's sustained `4.579955 tok/s` and therefore is not a successful final speed candidate.
- Quantitative blocker: accepted group-32 affine body storage is about 15.22 GB on the CPU. This Ryzen 7 250 sustains about 40-43 GB/s across the heterogeneous projection set, leaving roughly 235 ms/token in CPU INT4 work even after the largest safe GPU split. Lower-bit replacements fit and run faster but fail quality; lossless affine pages beyond the split do not fit beside the FP8 head and runtime buffers in 8 GB.
- Decision: cosine target achieved; speed target not achieved. Do not describe this run as beating Q4_K_M. Preserve the strongest validated profile and the failed speed evidence.

## 2026-07-14 final accounting and source audit

- Telemetry audit found that inactive decoder FP8 selectors were serialized as `"all"` even in a head-only FP8 run. Result reporting now emits `null` unless the corresponding down/QKV/O route is active; a five-case selector regression passed.
- The component report exposed the packed CPU INT4 body only through RSS and page-pool telemetry. It now records `packed_int4_weights_and_scales` as an explicit non-overlapping CPU owner. The accepted target cache is 15,220,556,336 bytes; a synthetic ownership regression totals page cache, pinned cache, packed INT4, and CPU KV exactly once.
- Derived-cache identity reconstruction from archive checksums plus calibration/covariance hashes found 496 accepted files (15,220,556,336 bytes) and 496 rejected experimental files (6,850,046,512 bytes). Only the rejected identities are marked for final deletion.
- Final static validation passed `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`, `cargo test --all-targets`, Python compileall, JSONL parsing, and `git diff --check`. The shared-tensor script explicitly skipped because no root-level default archive fixture exists and is not counted as a pass.

## 2026-07-14 matched 500-token closeout

- Restored Q4_K_M was exact-size and SHA-256 verified before use. A 99-layer forced server launch correctly OOMed because it disabled fit; the final disclosed configuration used `-ngl auto`, a 512-MiB fit target, one slot, F16 KV, and no prompt cache.
- Q4 exact-prompt runs retained all generated IDs: 200 tokens at 3.824263 tok/s and 500 tokens at 4.551783 tok/s. The latter reached 7,015 MiB, 59C, and 29.24W. An earlier `n_probs=1` run was rejected because per-token vocabulary ranking contaminated speed.
- ThinTensor matched 500-token public run reached 2.128140 tok/s, retained 512/512 causal BF16 KV positions with no miss/eviction/offload, and saved all 500 IDs. Peak telemetry was 6,500 MiB, 55C, and 18.18W. Corrected report fields show inactive decoder FP8 selectors as null and the 15,219,097,600-byte packed CPU body explicitly.
- Matched long-run verdict: ThinTensor is 53.25% slower and therefore fails the requested speed criterion. The centered-cosine win remains valid.
- Final cleanup deleted the twice-verified restored GGUF and 6,850,046,512 bytes of rejected cache variants. The 15,220,556,336-byte accepted cache and all evidence remain; final free space is 43,144,323,072 bytes.
