# ThinTensor final benchmark ledger

This report retains the apples-to-apples GPT-OSS control plus fresh Qwen3.5-9B, Llama 3.2 11B text-path, OpenReasoning-Nemotron-14B, and Phi-4 reruns. The cross-architecture operator matrix is complete in `benchmark_results/operator_graph_matrix.md`.

## Microsoft Phi-4 fresh result

The repaired installed public CLI selects `cpu-embed-mxfp4-mlp-int4g128-attention-head-full-resident` and completes the required 200/10 causal-KV loop at **28.2131 tok/s** (`28.1686` raw steady, 35.4445 ms/token) versus official same-model llama.cpp b9986 Q4_K at **17.4998 tok/s mean** across `17.5669 / 17.3769 / 17.5558`: **1.6122x, +61.22%**. All 40 layers are resident; the mode retains 7,209,625,600 weight bytes, peaks at 7,542,518,272 allocated / 7,635,730,432 reserved bytes and 65C, attends all 200 KV tokens, and reports `not_hf_equivalent=false`.

The speed and quality configurations are deliberately separate. The exact-BF16 streamed Thin quality mode has centered full-vocabulary cosine **0.999964 / 0.999958 / 0.999949** at identical factual steps 1/8/32 (minimum **0.999949**) versus official Q4_K **0.997595 / 0.997824 / 0.997281** (minimum **0.997281**). Exact Thin has raw minimum cosine 0.999956; both engines preserve top-1/top-5 at all checkpoints, and exact Thin matches all 32 teacher tokens. The low-bit speed mode itself is only experimental for quality (minimum centered cosine 0.983811) and is not misrepresented as the quality result.

The fresh official source revision is `932b33c0ec9ca189badeb22480721a8de9d0e006`; six verified BF16 shards total 29,319,042,992 file bytes. Public conversion produced a verified 243-page, 27.31-GiB archive with 29,319,014,400 raw tensor bytes and no unknown operators. Official `microsoft/phi-4-gguf` revision `6edc2ef6664b739a8e11e62f2672ff6afe0c15ac` provides the 9,053,114,560-byte Q4_K, SHA-256 `5652b9be0ea4ae2842130d04fe31bc869fcb99a2b7106c53b4e754a343fd688f`.

The speed fix is architecture-neutral: fused-QKV/fused-gate-up fallback now routes CPU embedding through the real GPU row-copy scratch path instead of indexing a metadata-only tensor. This removes the first-RMSNorm illegal access and allows the compact full-resident plan to participate in public auto-search.

Artifacts: `benchmark_results/phi4_fresh_final_public_200_10_v2.json`, `benchmark_results/phi4_fresh_final_search_v2/autofit_auto_search.json`, `benchmark_results/phi4_fresh_llamacpp_b9986_q4k_200_3.json`, `correctness_results/thin_modes/phi4_fresh_q4k_b9986_vs_source.json`, `correctness_results/thin_modes/phi4_fresh_exactbf16_stream_same5token.json`, and `correctness_results/thin_modes/phi4_fresh_speedmode_same5token.json`.

After those artifacts were verified nonempty, the 55-GiB source/archive cache and 8.5-GiB Q4 directory were deleted. Disk use returned from 83% to 56%.

## OpenReasoning-Nemotron-14B fresh result

The public ThinTensor CLI speed mode completes the required 200/10 causal-KV loop at **23.7869 tok/s** versus same-model llama.cpp b9986 Q4_K_M at **19.7889 tok/s mean** across three repetitions (**1.2020x**). The selected `cpu-embed-mxfp4-mlp-int4g128-attention-head-full-resident` profile uses exact CPU embedding-row fetch, MXFP4 MLP projections, group-128 INT4 attention, and a packed execution head. It retains 6,985,566,208 weight bytes, peaks at 7,420,454,912 CUDA allocated bytes and 58C, attends all 200 KV tokens, and reports `not_hf_equivalent=false`.

Quality is reported honestly as a separate exact-BF16 streamed ThinTensor mode, not as the speed configuration. Against the original BF16 source on the identical five-token trajectory and all vocabulary logits at steps 1/8/32, its centered cosine is **0.999602 / 0.999533 / 0.999686** (minimum **0.999533**) versus Q4_K_M **0.981811 / 0.967140 / 0.987023** (minimum **0.967140**); all top-1 checks pass.

The pinned official source revision `9e2805b596a173b5ef2f6d50c6fd9622135d2057` has three BF16 shards totaling 29,540,133,992 bytes and SHA-256 values `7bb855504d85ce5d8708d53916fcfadbdecd6a0c47c667960142d62b998d4844`, `9f9ac66e3e1537ad212e02b43565e5974a4acf3c0ab2b86154953b4b1b6cd5af`, and `41f8ea22b534d3390660ab7778e23ca663b610ec27ef51900b4f78f572cac04b`. The verified Thin archive has 579 pages and is 27.51 GiB. The pinned 8,988,108,960-byte Q4_K_M SHA-256 is `e47e8a6b37cfcbe4f2f89179d2e628a29f0c05643a4d06dbca6588daf91e407c`.

Artifacts: `benchmark_results/nemotron14b_fresh_final_public_200_10_v3.json`, `benchmark_results/nemotron14b_fresh_final_search_v3/autofit_auto_search.json`, `benchmark_results/nemotron14b_fresh_llamacpp_b9986_q4km_200_3.json`, `correctness_results/thin_modes/nemotron14b_fresh_exactbf16_stream_same5token.json`, and `correctness_results/thin_modes/nemotron14b_fresh_q4km_b9986_vs_source.json`.

## Llama 3.2 11B Vision fresh text-path result

ThinTensor's quality-qualified hybrid reaches **41.0158 tok/s** on the required 200/10 causal-KV run versus same-source llama.cpp b9986 Q4_K_M at **39.8206 tok/s** across three repetitions (**1.0300x**). At the identical six-token factual prompt and BF16-teacher checkpoints 1/8/32, ThinTensor's minimum raw full-vocabulary cosine is **0.996272** and minimum centered cosine is **0.995095**, both above Q4_K_M's centered **0.991938**; all top-1 checks pass.

The final profile keeps attention and MLP layers 0–27 in scaled FP8 tensor-core form, uses refined group-8 INT4 only for the final four MLP layers, and keeps the embedding/head in FP8. It retains 7,292,962,848 weight bytes, attends all 200 causal-KV tokens, reports `not_hf_equivalent=false`, and runs at 24.3808 ms/token. The optimization added FP16 grouped scales, least-squares INT4 scale refinement, role/layer-specific group controls, a CPU-reference device option for the llama.cpp full-logit comparator, and an auto-search candidate for this resident quality profile.

The source's five pinned shards total 21,340,560,870 bytes and match the SHA-256 values recorded in `OVERNIGHT_SCRATCHPAD.md`. The text archive has 291 pages, 32 compacted self-attention layers, and is 14.96 GiB. The exact same-source text-only Q4_K_M GGUF contains 8,030,261,312 parameters, is 4,912,898,304 bytes, and has SHA-256 `17e4d9e4b25c1480f90da29b2ee019734200f82c00e71f22beafe6903fded236`.

Artifacts: `benchmark_results/llama32_fresh_final_quality_speed_200_10.json`, `benchmark_results/llama32_fresh_llamacpp_b9986_q4km_200_3.json`, `correctness_results/thin_modes/llama32_fresh_final_fp8_mlp0_28_int4g8_tail_fp8attn_same6token.json`, and `correctness_results/llama32_fresh_q4km_b9986_vs_source.json`.

## Qwen3.5-9B fresh b9986 result

ThinTensor passes the requested same-model Q4 gate on the RTX 5050 Laptop GPU: **43.2832 tok/s** versus llama.cpp b9986 Q4_K_M **42.8302 tok/s mean** (**1.0106x, +1.06%**). On identical factual prompt tokens and BF16 teacher checkpoints at steps 1/8/32, ThinTensor's minimum full-vocabulary cosine is **0.973334** versus Q4_K_M **0.894908**. Both preserve top-1; Thin's minimum top-5 overlap is 0.8 versus Q4's 0.4.

### Setup and provenance

- Hardware: NVIDIA GeForce RTX 5050 Laptop GPU, 8,151 MiB VRAM; batch 1.
- Source: fresh `Qwen/Qwen3.5-9B` pull. Safetensor sizes are 5,276,436,216 / 5,335,161,512 / 5,368,717,440 / 3,325,995,712 bytes. SHA-256 values are `db6f444b43d318c92f360a13a25561a6a65b10c0631b8ed305a426dbaa6c380e`, `31c7d7e2dd5d207840b31cc59083c8f4c4718959149e0358c0364052bb9a0330`, `7ec36ba3a4176a44c3c0876ad80c56a2f70c84bf008d82e9501df642f17dadec`, and `b62b0c4cd7e44edee103ee8f4fe225f246d5e768e07bfd5f25b63a8aa1fdd0c6`; all match Hugging Face LFS metadata.
- Conversion: `qwen-9b-fresh.thin`, 427 pages, 32 text layers, 16.68 GiB. Conversion excluded 348 vision/MTP tensors and passed archive verification.
- GGUF: `unsloth/Qwen3.5-9B-GGUF`, revision `3885219b6810b007914f3a7950a8d1b469d598a5`, `Qwen3.5-9B-Q4_K_M.gguf`, 5,680,522,464 bytes, SHA-256 `03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8`.
- llama.cpp: official prebuilt b9986, commit `91c631b21`, Vulkan1 RTX 5050, all 99 layers requested on GPU, batch/ubatch 1, F16 KV, 512 MiB fit margin, 2,048-token minimum fit context.

### Throughput

| Engine | Measured shape | Result | Range |
|---|---|---:|---:|
| ThinTensor public CLI | 10 warmup + 200 measured | **43.2832 tok/s**, 23.1197 ms/token | peak 64C |
| llama.cpp Q4_K_M | 200 generated, 3 repetitions | **42.8302 tok/s mean** | 53.1848 / 41.5511 / 33.7546 |

The selected public autofit candidate is `recurrent-mxfp4-mlp-int4g512-attention-full-resident`: fused dual-MXFP4 gate/up projection, MXFP4 MLP, group-512 INT4 attention matrices, fused Triton causal attention, tuned large-matvec dispatch, FP8 embedding/head, BF16 activations/KV, and an 8,000,000,000-byte weight budget.

### Full-vocabulary quality

The shared factual prompt has five tokens. Both engines follow the same BF16 teacher trajectory and compare all 248,320 vocabulary entries at steps 1, 8, and 32. Cross-runtime cosine is centered to remove llama.cpp's unknown additive log-softmax constant.

| Engine | Step 1 | Step 8 | Step 32 | Minimum | Min top-5 | Top-1 |
|---|---:|---:|---:|---:|---:|:---:|
| ThinTensor optimized | 0.988964 | 0.973334 | 0.981885 | **0.973334** | **0.8** | all pass |
| llama.cpp Q4_K_M | 0.995091 | 0.986902 | 0.894908 | 0.894908 | 0.4 | all pass |

Q4's maximum Jensen-Shannon distance is 0.057975 and maximum total-variation distance is 0.062302. ThinTensor's generated-token match rate is 0.96875 on the recorded 32-step factual trajectory.

### ThinTensor telemetry

- Runtime: `recurrent_hybrid_decoder`; schema `heterogeneous:gated_delta_net_dense,separate_qkv_gated_dense`; causal KV; `kv_tokens_attended=200`; `not_hf_equivalent=false`; no approximation reason.
- Memory: 4,647,777,280 resident weight bytes; 5,732,285,952 peak CUDA allocated; 5,746,196,480 peak CUDA reserved; 5,713,802,240 startup transfer bytes.
- Timing: 4.5976 seconds steady decode; 2.7531 ms/token vocabulary head; 20.3556 ms/token measured forward component time; 35.8115 seconds load.
- KV: exact BF16 GPU-resident cache, 131,072 bytes/token, 819,200 bytes read at the final step, no KV H2D/D2H, compression, or eviction.
- Static launch estimate: 322 launches/token (129 matvec, 96 attention, 97 elementwise). The new dual-MXFP4 kernel combines equal-shape gate/up work in one grid; the static estimator has not yet been made operator-graph aware and is retained as emitted rather than manually altered.

### Qwen artifacts

- `benchmark_results/qwen35_fresh_b9986_optimized_200_10.json`
- `benchmark_results/qwen35_fresh_b9986_optimized_search/autofit_auto_search.json`
- `benchmark_results/qwen35_fresh_llamacpp_b9986_q4km_200_3.json`
- `benchmark_results/qwen35_fresh_b9986_speed_quality_comparison.json`
- `correctness_results/thin_modes/qwen35_fresh_optimized_group512_factual_prefill5.json`
- `correctness_results/thin_modes/qwen35_fresh_optimized_group512_steps1_8_32.json`
- `correctness_results/qwen35_fresh_q4km_b9986_vs_source.json`

### Reproduction

```bash
thintensor bench ~/.cache/thintensor/models/Qwen--Qwen3.5-9B/qwen-9b-fresh.thin --profiles autofit --steps 200 --warmup 10 --auto-search --auto-search-steps 4 --auto-search-warmup 1 --max-gpu-temp 83 --auto-search-out benchmark_results/qwen35_fresh_b9986_optimized_search --out benchmark_results/qwen35_fresh_b9986_optimized_200_10.json

/tmp/llama-b9986-vulkan/llama-b9986/llama-bench -m benchmark_models/qwen3.5-9b-gguf/Qwen3.5-9B-Q4_K_M.gguf -p 1 -n 200 -b 1 -ub 1 -ngl 99 -dev Vulkan1 -fitt 512 -fitc 2048 -r 3 -o json
```

## GPT-OSS-20B retained control

The existing accepted result remains: ThinTensor's full-resident middle-out Q1/Q2 path uses exact embedding/head, full router top-4, causal KV, and explicit Torch attention. Its repeated 200-token median is **50.3445 tok/s** versus fresh llama.cpp b9964 Q4_K_M **40.3631 tok/s mean** (**1.2473x, +24.73%**). Source-relative ThinTensor centered-logit cosine is 0.987376 at step 1 and 0.983299 at step 4 with top-1 preserved; llama.cpp Q4_K_M step-1 centered cosine is 0.792350. GPT-OSS evidence remains preserved as directed.


Disclaimer: this file was written by a codex, while running a benchmark i asked it to do, this was ran overnight, under arch performance mode
