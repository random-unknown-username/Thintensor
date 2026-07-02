# Benchmark matrix

GPU: NVIDIA GeForce RTX 5050 Laptop GPU, 8 GB. Throughput is batch-1
single-token decode.

| Model | Mode | Attention | Steps | tok/s | Correctness | Worst cosine | Notes |
|:---|:---|:---|---:|---:|:---|---:|:---|
| SmolLM3-3B | BF16 | causal KV | 300 | 31.89 | safe default | 0.99932 exact reference | `triton-matvec`; strict eager-HF exact gate passes |
| SmolLM3-3B | gate/up FP8 + down FP8 8:28 | causal KV | 500 | 46.41 | ranking pass, opt-in | 0.99804 required; 0.99379 stress | quality-first selective FP8; 2,070,749,184 bytes saved |
| SmolLM3-3B | gate/up FP8 + down FP8 6:30 + O FP8 4:32 | causal KV | 500 | 49.88 | ranking pass, experimental | 0.99650 required; 0.98471 stress | faster but larger chat-template drift |
| SmolLM3-3B | full MLP FP8 | causal KV | 100 | 50.61 | ranking pass, experimental | 0.98378 | selective body quantization |
| SmolLM3-3B | full MLP + O FP8 | causal KV | 100 | 51.53 | ranking pass, experimental | 0.97096 | selective body quantization |
| Qwen3-0.6B | Head8 | causal KV | 100 | 97.93 | correctness not rerun | — | archive no longer present; result retained from JSON |
| Qwen3-0.6B | Head8 smoke | current token only | 100 | 191.47 | not HF-equivalent | — | smoke result only; skips historical KV |
| Qwen3-4B | BF16 stream/offload | causal KV | — | — | not rerun | — | archive absent |

Source JSON is in `benchmark_results/` and `correctness_results/`. Missing
models are reported as unavailable instead of borrowing results from a
different precision, attention, or residency mode.
