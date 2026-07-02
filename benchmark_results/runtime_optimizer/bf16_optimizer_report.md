# Runtime optimization report

Every candidate ran in isolated processes. Accepted candidates become the baseline for the next candidate; rejected and experimental candidates are not stacked.

| Candidate | Speed change | GPU memory change | KV memory change | Min cosine change | Top1/top5 impact | Decision |
|---|---:|---:|---:|---:|---|---|
| bf16_lm_head_fp8_shortlist_bf16_verify | +3.46% | +263179776 | +0 | +0.000000 | required failures 0, stress failures 1, min top5 0.800 | accept |

- **bf16_lm_head_fp8_shortlist_bf16_verify hypothesis:** A separate FP8 execution head can shortlist candidates while BF16 rows preserve exact final selection
- Implementation: Shortlist 64 IDs with row-scaled FP8 and verify them with the resident BF16 tied head
- Expected bottleneck: BF16 LM-head weight bandwidth
- Decision reason: all speed and correctness gates passed
- Enabled by default: `false`
- Benchmark command: `/usr/bin/python3 /home/satvik/Projects/thintensor-opus/scripts/thin_runtime.py run SmolLM3-3B.thin --device cuda --dtype bf16 --steps 500 --warmup-steps 10 --residency all --kernel-backend triton --attention-mode causal_kv --lm-head-backend triton --lm-head-fp8 --keep-bf16-lm-head --lm-head-topk-guard 64 --json`
- Correctness command: `/usr/bin/python3 /home/satvik/Projects/thintensor-opus/scripts/compare_hf_thin_logits.py --hf-model SmolLM3-3B --archive SmolLM3-3B.thin --device cuda --dtype bf16 --prompt Hello --prefill-lens 1,8,128 --steps 1,10 --kernel-backend triton --lm-head-backend triton --lm-head-fp8 --lm-head-topk-guard 64 --out /home/satvik/Projects/thintensor-opus/correctness_results/runtime_optimizer/bf16_lm_head_fp8_shortlist_bf16_verify_required.md --json`

| bf16_lm_head_persistent_vocab_block | +0.12% | +0 | +0 | n/a | correctness skipped after speed failure | reject |

- **bf16_lm_head_persistent_vocab_block hypothesis:** A single-stage, persistent vocab-block argmax Triton kernel can reduce intermediate memory traffic and launch overhead compared to two-stage argmax, while maintaining BF16 precision and exact matches.
- Implementation: Launch a persistent JIT grid that loops over vocab blocks, maintains local max/index, and performs grid reduction via an atomic counter in a single kernel.
- Expected bottleneck: BF16 LM-head two-stage Triton kernel overhead
- Decision reason: 200-token speed change -0.70% is below +2.00%; 500-token speed change +0.12% is below +2.00%
- Enabled by default: `false`
- Benchmark command: `/usr/bin/python3 /home/satvik/Projects/thintensor-opus/scripts/thin_runtime.py run SmolLM3-3B.thin --device cuda --dtype bf16 --steps 500 --warmup-steps 10 --residency all --kernel-backend triton --attention-mode causal_kv --lm-head-backend triton --lm-head-fp8 --keep-bf16-lm-head --lm-head-topk-guard 64 --lm-head-argmax-mode triton_persistent --json`
- Correctness command: not run because the candidate failed the speed gate.


## Locked baseline and correctness

- 200-token steady speed: `32.902 tok/s`
- 500-token steady speed: `32.461 tok/s`
- Required matrix: `0` top-1 failures, minimum top-5 `1.000`, minimum cosine `0.997313`.
- 13-case stress: `1` checkpoint top-1 failure, minimum top-5 `0.800`, minimum cosine `0.999109`, maximum TV `0.030315`, maximum JS `0.003072`.

## Rejected hot-path controls

| Candidate | Isolated result | Full decode result | Decision |
|---|---|---|---|
| Fused residual + RMSNorm | bit-exact; 0.01649 ms to 0.00885 ms; old logical launch count 434 to 362 | approximately -0.17% at 500 tokens | reject |
| Fused scaled gate/up + SiLU | bit-exact; MLP component about 5-6% faster | approximately +0.04% at 500 tokens | reject |
| Interleaved gate/up storage | max abs 0.000122; about 3.8x slower | not promoted | reject |
| Two-stream gate/up | bit-exact | component about 18.3% slower | reject |
| Pipeline-stage tuning | bit-exact | no stable randomized A/B win | reject |

These controls show why launch count and isolated tiny-kernel latency are not used as decode-win evidence.

## Bottleneck table

| Component | Measured ms | Note |
|---|---:|---|
| LM-head matvec | 1.547 | standalone post-decode component profile |
| Argmax | 0.041 | standalone post-decode component profile |
| Per-layer attention | 0.328 | profiler event instrumentation adds overhead |
| Per-layer MLP | 0.291 | gate/up/down plus activation |
| Per-layer O projection | 0.047 | included in attention total |

The profiler component pass is diagnostic and runs after the timed decode loop; its event overhead means component totals must not be substituted for the real 500-token rate.

## Exact KV residency curve at 512 tokens

| Mode | Steady tok/s | GPU KV bytes | CPU KV bytes | H2D bytes | Decision |
|---|---:|---:|---:|---:|---|
| gpu_full | 48.079 | 37748736 | 0 | 0 | baseline |
| hybrid_recent | 46.048 | 18874368 | 18874368 | 2283798528 | exact opt-in; not a speed optimization |
| cpu_exact | 41.298 | 1179648 | 36569088 | 9361686528 | exact opt-in; not a speed optimization |

## Outcome

- Best accepted runtime optimization: `bf16_lm_head_fp8_shortlist_bf16_verify`.
- Best rejected idea: `bf16_lm_head_persistent_vocab_block` (200-token speed change -0.70% is below +2.00%; 500-token speed change +0.12% is below +2.00%).
- Next bottleneck: improve bytes-per-second in the large MLP down, gate/up, O-proj, and QKV matvecs. Launch-only fusion is not the current priority.
- Exact KV memory reduction: yes for opt-in residency. At 512 tokens, `cpu_exact` reduced resident GPU KV from 37,748,736 to 1,179,648 bytes while preserving total BF16 KV bytes.
- KV compression: not tested. Exact layout/residency tiers were completed first; no lossy KV mode is enabled.
