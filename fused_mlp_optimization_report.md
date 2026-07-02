# Fused MLP optimization report

## Verdict

No fused MLP implementation passed the 2% real-decode retention gate on the
current RTX 5050 Laptop GPU. Fused MLP remains disabled by default.

The retained quality-first configuration instead uses scaled row-wise FP8 for
gate/up in every layer and down projection in layers 8:28. It measured 46.41
tok/s over 500 causal-KV tokens, saved 2,070,749,184 resident body-weight
bytes, and reached minimum required-test cosine 0.99804.

## 2026-07-01 bottleneck follow-up

| Candidate | Result | Decision |
|:---|:---|:---|
| two-stage BF16 LM-head argmax | 47.14 tok/s vs 47.79 retained | reject |
| fused single-token GQA attention | 47.7943 tok/s vs 47.7949 retained | reject |
| quality FP8 body + Head8 | 50.44 tok/s | experimental; long-case top-1 drift |

The retained quality mode keeps its BF16 execution head.

## Measured experiments

| Experiment | Baseline | Candidate | Delta | Decision |
|:---|---:|---:|---:|:---|
| Fused FP8 gate/up plus SiLU | 48.11 tok/s | 48.39 tok/s | about +0.6% | Reject; below 2% |
| Fused residual plus RMSNorm | 51.05 tok/s | 50.36 tok/s | regression | Reject |
| Runtime QKV/gate-up concatenation | small isolated gain | increased resident memory substantially | not a real-decode win | Keep opt-in only |
| Existing `--fused-mlp` path | no validated end-to-end win | no default change | — | Disabled |

Raw benchmark JSON is under:

- `benchmark_results/fused_fp8_gate_up/`
- `benchmark_results/fused_residual_norm/`
- `benchmark_results/role_backend_grid/`

The remaining MLP cost in the quality-first 500-token profile is about
0.287 ms per layer. The tested launch fusions did not overcome their added
dispatch/layout overhead. A future attempt should use a packed native FP8 dot
layout and fuse scale application, activation, and down-projection traffic as a
single design; another wrapper around the current kernels is unlikely to clear
the retention gate.
