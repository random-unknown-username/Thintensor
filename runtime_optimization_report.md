# Runtime optimization report

Every candidate ran in isolated processes. Accepted candidates become the baseline for the next candidate; rejected and experimental candidates are not stacked.

| Candidate | Speed change | GPU memory change | KV memory change | Min cosine change | Top1/top5 impact | Decision |
|---|---:|---:|---:|---:|---|---|
| lm_head_fp8_shortlist_bf16_verify | +4.60% | +263181312 | +0 | +0.000000 | required failures 0, stress failures 0, min top5 0.800 | accept |

- **lm_head_fp8_shortlist_bf16_verify hypothesis:** Reading an FP8 execution head and recomputing only shortlisted logits in BF16 can halve the dominant LM-head weight traffic without changing public logits or final candidate math
- Implementation: Use row-scaled FP8 only to shortlist 64 vocabulary IDs, recompute those rows from the resident tied BF16 head, and preserve BF16 argmax tie-breaking
- Expected bottleneck: BF16 LM-head weight bandwidth
- Decision reason: all speed and correctness gates passed
- Enabled by default: `false`
- Benchmark command: `/usr/bin/python3 /home/satvik/Projects/thintensor-opus/scripts/thin_runtime.py run SmolLM3-3B.thin --device cuda --dtype bf16 --steps 500 --warmup-steps 10 --residency all --kernel-backend triton --attention-mode causal_kv --lm-head-backend triton --gate-up-fp8 --down-proj-fp8 --down-fp8-layers 8:28 --lm-head-fp8 --keep-bf16-lm-head --lm-head-topk-guard 64 --json`
- Correctness command: `/usr/bin/python3 /home/satvik/Projects/thintensor-opus/scripts/compare_hf_thin_logits.py --hf-model SmolLM3-3B --archive SmolLM3-3B.thin --device cuda --dtype bf16 --prompt Hello --prefill-lens 1,8,128 --steps 1,10 --kernel-backend triton --lm-head-backend triton --gate-up-fp8 --down-proj-fp8 --down-fp8-layers 8:28 --lm-head-fp8 --lm-head-topk-guard 64 --out /home/satvik/Projects/thintensor-opus/correctness_results/runtime_optimizer/lm_head_fp8_shortlist_bf16_verify_required.md --json`

| split_k_down_projection | +1.69% | +0 | +0 | n/a | correctness skipped after speed failure | reject |

- **split_k_down_projection hypothesis:** Two exact column partitions provide enough CTAs to saturate the low-row down-projection GEMV
- Implementation: Compute two FP32 partial sums per output row and reduce them into the existing BF16 output
- Expected bottleneck: BF16 and scaled-FP8 down-projection bandwidth
- Decision reason: 200-token speed change +1.34% is below +2.00%; 500-token speed change +1.69% is below +2.00%
- Enabled by default: `false`
- Benchmark command: `/usr/bin/python3 /home/satvik/Projects/thintensor-opus/scripts/thin_runtime.py run SmolLM3-3B.thin --device cuda --dtype bf16 --steps 500 --warmup-steps 10 --residency all --kernel-backend triton --attention-mode causal_kv --lm-head-backend triton --gate-up-fp8 --down-proj-fp8 --down-fp8-layers 8:28 --lm-head-fp8 --keep-bf16-lm-head --lm-head-topk-guard 64 --split-k-down-proj --json`
- Correctness command: not run because the candidate failed the speed gate.


## Final current-source standalone checks

| Path | Steady tok/s | ms/token | Resident weights | Read bytes/token | Peak allocated |
|---|---:|---:|---:|---:|---:|
| BF16 | 34.310 | 29.136 | 6150197248 | 6149898240 | 6313040896 |
| BF16 guarded | 35.206 | 28.390 | 6413378560 | 5888005120 | 6576220672 |
| Quality | 48.625 | 20.556 | 4076113920 | 4075814912 | 6304497664 |
| Quality guarded | 50.884 | 19.645 | 4339295232 | 3813921792 | 6567678976 |

These are final-state standalone 500-token checks. Balanced ABBA results above control acceptance because laptop power and thermals shift absolute standalone rates.

## Locked baseline and correctness

- 200-token steady speed: `48.798 tok/s`
- 500-token steady speed: `48.170 tok/s`
- Required matrix: `0` top-1 failures, minimum top-5 `1.000`, minimum cosine `0.998661`.
- 13-case stress: `0` checkpoint top-1 failure, minimum top-5 `0.800`, minimum cosine `0.991255`, maximum TV `0.086904`, maximum JS `0.016176`.

## Accepted guarded LM-head paths

The FP8 execution head only selects 64 candidate IDs. Candidate logits, public full logits, and final tie-breaking remain BF16.

| Path | Tokens | Baseline tok/s | Guarded tok/s | Delta |
|---|---:|---:|---:|---:|
| Quality FP8 | 200 | 48.486 | 51.026 | +5.24% |
| Quality FP8 | 500 | 47.608 | 49.798 | +4.60% |
| Full BF16 | 200 | 31.533 | 32.689 | +3.67% |
| Full BF16 | 500 | 31.356 | 32.441 | +3.46% |

| Path | Baseline resident weights | Guarded resident weights | Baseline read bytes/token | Guarded read bytes/token |
|---|---:|---:|---:|---:|
| Quality FP8 | 4076113920 | 4339295232 | 4075814912 | 3813921792 |
| Full BF16 | 6150197248 | 6413378560 | 6149898240 | 5888005120 |

The guarded path adds `263181312` resident bytes and removes `261893120` estimated head-read bytes per token. It remains explicit rather than universal because shortlist recall is model/profile specific.

Full-BF16 required and stress metrics were unchanged: required minimum cosine `0.997313`, stress minimum cosine `0.999109`, and stress minimum top-5 `0.800`.

## Rejected hot-path controls

| Candidate | Isolated result | Full decode result | Decision |
|---|---|---|---|
| Fused residual + RMSNorm | bit-exact; 0.01649 ms to 0.00885 ms; old logical launch count 434 to 362 | approximately -0.17% at 500 tokens | reject |
| Fused scaled gate/up + SiLU | bit-exact; MLP component about 5-6% faster | approximately +0.04% at 500 tokens | reject |
| Interleaved gate/up storage | max abs 0.000122; about 3.8x slower | not promoted | reject |
| Two-stream gate/up | bit-exact | component about 18.3% slower | reject |
| Pipeline-stage tuning | bit-exact | no stable randomized A/B win | reject |
| Persistent FP8 gate/up variants | isolated kernels improved about 21-24% | best full decode gain remained below 2% | reject |
| Persistent BF16 gate/up | isolated kernel improved about 11% | full decode was flat | reject |
| Lossless contiguous QKV packing | bit-exact | +0.41% full decode | reject |
| Tensor-core skinny GEMM | numerically valid prototype | slower than the bandwidth GEMV | reject |
| Context-aware split-head attention | microkernel cosine about 0.99998 | -1.65% at 1000-token decode | reject |
| QKV + O FP8 12:24 | stress top-1 recovered | +0.05% versus simpler O-only profile | reject |
| Row-swizzled FP8 down | repeated-weight kernel up to 1.88x; 500-token +2.84% | introduced sampling_3 top-1 failures at steps 16 and 64 | reject |
| Arithmetic-preserving row schedule | exact original reduction groups | +1.35% at 200 tokens | reject |
| Paired 128-column down loads | bit-exact microkernel 1.76x | +0.50% at 200 tokens | reject |
| FP16 packed products, FP32 accumulation | kernel cosine 0.9999997 | -0.80% at 200 tokens | reject |
| On-chip K-chunk MLP pipeline | bit-exact at best tile | best kernel remained 28% slower | reject |
| Static column-tiled BF16 head | +525336576 resident bytes; cosine about 1.0 | 1.7490 ms to 2.0017 ms | reject |

These controls show why launch count and isolated tiny-kernel latency are not used as decode-win evidence.

## Ranked bottleneck table

| Component | BF16 ms/token | Quality ms/token | BF16 % | Quality % | Read bytes BF16 / quality | Launches BF16 / quality | Candidate optimization |
|---|---:|---:|---:|---:|---:|---:|---|
| Gate/up projections | 11.009 | 5.428 | 39.7% | 24.1% | 3246391296 / 1626365952 | 36 / 72 | persistent FP8 GEMV, role-specific row scheduling |
| Down projection | 6.031 | 3.674 | 21.7% | 16.3% | 1623195648 / 1172471808 | 36 / 36 | loop-tile tuning, split-K only if end-to-end wins |
| Attention core | 1.999 | 3.169 | 7.2% | 14.1% | 1024000 / 1024000 | 108 / 108 | context-aware block-split attention at long context |
| QKV projections | 1.981 | 1.528 | 7.1% | 6.8% | 452984832 / 452984832 | 36 / 36 | net-zero contiguous packing, persistent grouped GEMV |
| LM head + argmax | 1.763 | 1.588 | 6.4% | 7.0% | 525336576 / 525336576 | 2 / 2 | guarded shortlist, persistent vocab reduction, prepacked head |
| O projection | 1.352 | 1.695 | 4.9% | 7.5% | 301989888 / 301989888 | 36 / 36 | role-specific GEMV, exact layout packing |

The profiler component pass is diagnostic and runs after the timed decode loop; its event overhead means component totals must not be substituted for the real 500-token rate.

## Exact KV residency curve at 512 tokens

| Mode | Steady tok/s | GPU KV bytes | CPU KV bytes | H2D bytes | Decision |
|---|---:|---:|---:|---:|---|
| gpu_full | 48.079 | 37748736 | 0 | 0 | baseline |
| hybrid_recent | 46.048 | 18874368 | 18874368 | 2283798528 | exact opt-in; not a speed optimization |
| cpu_exact | 41.298 | 1179648 | 36569088 | 9361686528 | exact opt-in; not a speed optimization |

## Exact BF16 weight residency curve

These are same-command 10-token capacity measurements, not steady 500-token speed claims. CPU offload preserves BF16; it is a capacity feature and is not called faster.

| Budget | Steady tok/s | GPU resident weights | CPU resident weights | H2D bytes/token | Prefetch overlap | Correctness |
|---|---:|---:|---:|---:|---:|---|
| 1GB | 2.624 | 681586688 | 6150197248 | 5499591773 | 0.005 | top1 failures 0; top5 1.0; cosine 0.997313 |
| 2GB | 3.146 | 1619062784 | 6150197248 | 4604728227 | 0.001 | top1 failures 0; top5 1.0; cosine 0.997313 |
| 4GB | 5.658 | 3650260992 | 6150197248 | 2665857210 | 0.005 | top1 failures 0; top5 1.0; cosine 0.997313 |
| 6GB | 28.139 | 5681459200 | 6150197248 | 726986193 | 0.037 | top1 failures 0; top5 1.0; cosine 0.997313 |
| all | 45.347 | 6150197248 | 0 | 0 | 0.000 | all-resident reference |

The page pool pins a deterministic whole-layer prefix and reserves the current/prefetched working set. This avoids the cyclic-LRU failure mode that reloaded nearly every layer even at a 6 GB budget. Backend dispatch now uses stable shape/page-role keys rather than recycled tensor object ids.

## Guarded LM-head shortlist recall

| K | Top-1 recall | Complete top-5 recall |
|---:|---:|---:|
| 32 | 1.000 | 1.000 |
| 64 | 1.000 | 1.000 |
| 128 | 1.000 | 1.000 |
| 256 | 1.000 | 1.000 |
| 512 | 1.000 | 1.000 |
| 1024 | 1.000 | 1.000 |

Retained K: `64`. K=32 improved balanced 200-token decode by only 0.09% versus K=64, below the 2% gate; retain the larger safety margin.

## Attention context crossover audit

| Context | Existing torch ms | Fused ms | Fused / torch | Cosine |
|---:|---:|---:|---:|---:|
| 512 | 0.055850 | 0.038029 | 0.68x | 0.999983 |
| 1024 | 0.054855 | 0.067463 | 1.23x | 0.999982 |
| 2048 | 0.056124 | 0.133589 | 2.38x | 0.999986 |
| 4096 | 0.055889 | 0.266154 | 4.76x | 0.999983 |

The fused kernel wins only in the isolated 512-token case, loses increasingly from 1024 through 4096, and the context-aware full-decode switch regressed the 1000-token loop by 1.65 percent.
These are isolated diagnostics; the full-decode regression controls the decision.

## Outcome

- Best accepted runtime optimization: `lm_head_fp8_shortlist_bf16_verify`.
- Default settings: unchanged. The guarded LM-head path remains an explicit SmolLM3-validated profile because it carries a 263,181,312-byte execution copy and shortlist recall is model-specific.
- Best rejected idea: `row_swizzled_fp8_down` (+2.84% at 500 tokens, but it introduced sampling_3 checkpoint top-1 failures at steps 16 and 64).
- Next bottleneck: reduce real large-projection weight bytes or make exact FP8 conversion/reduction cheaper under the 35 W power ceiling. Launch-only and repeated-weight scheduler wins have been exhausted.
- Exact KV memory reduction: yes for opt-in residency. At 512 tokens, `cpu_exact` reduced resident GPU KV from 37,748,736 to 1,179,648 bytes while preserving total BF16 KV bytes.
- KV compression: not tested. Exact layout/residency tiers were completed first; no lossy KV mode is enabled.
