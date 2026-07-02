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


## Locked baseline and correctness

- 200-token steady speed: `48.798 tok/s`
- 500-token steady speed: `48.170 tok/s`
- Required matrix: `0` top-1 failures, minimum top-5 `1.000`, minimum cosine `0.998661`.
- 13-case stress: `0` checkpoint top-1 failure, minimum top-5 `0.800`, minimum cosine `0.991255`, maximum TV `0.086904`, maximum JS `0.016176`.

## Bottleneck table

| Component | Measured ms | Note |
|---|---:|---|
| LM-head matvec | 1.459 | standalone post-decode component profile |
| Argmax | 0.025 | standalone post-decode component profile |
| Per-layer attention | 0.365 | profiler event instrumentation adds overhead |
| Per-layer MLP | 0.292 | gate/up/down plus activation |
| Per-layer O projection | 0.290 | included in attention total |

The profiler component pass is diagnostic and runs after the timed decode loop; its event overhead means component totals must not be substituted for the real 500-token rate.

## Exact KV residency curve at 512 tokens

| Mode | Steady tok/s | GPU KV bytes | CPU KV bytes | H2D bytes | Decision |
|---|---:|---:|---:|---:|---|
| gpu_full | 48.079 | 37748736 | 0 | 0 | baseline |
| hybrid_recent | 46.048 | 18874368 | 18874368 | 2283798528 | exact opt-in; not a speed optimization |
| cpu_exact | 41.298 | 1179648 | 36569088 | 9361686528 | exact opt-in; not a speed optimization |

## Outcome

- Best accepted runtime optimization: `lm_head_fp8_shortlist_bf16_verify`.
- Best rejected idea: `split_k_down_projection` (200-token speed change +1.34% is below +2.00%; 500-token speed change +1.69% is below +2.00%).
- Next bottleneck: replace Python page traversal and repeated improve bytes-per-second in the large MLP down, gate/up, O-proj, and QKV matvecs. Launch-only fusion is not the current priority.
- Exact KV memory reduction: yes for opt-in residency. At 512 tokens, `cpu_exact` reduced resident GPU KV from 37,748,736 to 1,179,648 bytes while preserving total BF16 KV bytes.
- KV compression: not tested. Exact layout/residency tiers were completed first; no lossy KV mode is enabled.
