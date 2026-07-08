# Benchmark matrix

Measurements below were collected on 2026-07-05 with an NVIDIA GeForce RTX
5050 Laptop GPU. Every speed result uses one decode stream, full causal
attention, 10 warmup tokens, and at least 200 measured tokens. Transformers
baselines use the same local model, BF16 dtype, device, and token count.

Peak memory is CUDA peak allocated memory, not model-file size. A positive
memory delta means ThinTensor used more VRAM than Transformers. The guarded
FP8 head and adaptive INT8 body trade additional resident acceleration
structures for bandwidth and preserve a BF16 source or exact shortlist guard.

## Fastest correctness-tested profile by model

| Model | Status | Tokens | Thin tok/s | HF tok/s | Speedup | Thin/HF peak GiB | VRAM delta | Min cosine | Top-1 | Top-5 set | Top-5 order | KV retention |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|:---:|:---|
| SmolLM3-3B | verified | 500 | 94.00 | 47.54 | 1.977x | 6.124 / 5.878 | +4.2% | 0.997370 | yes | yes | no | full BF16 |
| StableLM-3B-4E1T | verified | 200 | 55.76 | 48.28 | 1.155x | 5.311 / 5.334 | -0.4% | 0.998577 | yes | yes | no | full BF16 |
| Phi-4-mini-instruct | verified | 200 | 53.80 | 39.09 | 1.376x | 7.223 / 7.180 | +0.6% | 0.999904 | yes | yes | yes | full BF16 |
| Qwen2.5-3B-Instruct | verified | 500 | 93.08 | 48.18 | 1.932x | 6.189 / 5.871 | +5.4% | 0.998892 | yes | yes | yes | full BF16 |
| Gemma-2-2B-IT | verified | 200 | 57.29 | 53.58 | 1.069x | 5.463 / 4.901 | +11.5% | 0.999555 | yes | yes | yes | full BF16 |
| TinyLlama-1.1B-Chat | candidate | 200 | 180.28 | 132.36 | 1.362x | 2.137 / 2.064 | +3.5% | 0.997788 | yes | no | no | full BF16 |
| Qwen3.5-0.8B | candidate | 200 | 115.91 | 64.33 | 1.802x | 1.453 / 1.452 | +0.1% | 0.999811 | yes | no | no | full BF16 |
| Gemma-4-E2B | candidate | 200 | 43.60 | 1.62 | 26.91x | 6.490 / 6.620 | -2.0% | 0.999671 | yes | yes | no | full BF16 |
| OLMoE-1B-7B-0924-Instruct | candidate | 200 | 33.70 | 9.22 | 3.65x | 6.820 / 6.450 | +5.7% | 0.999218 | yes | yes | no | full BF16 |

`candidate` is deliberate for TinyLlama: the short suite retained exact top-1
and 0.997788 minimum cosine, but one BF16 fifth-place cutoff tie changed the
strict top-5 set. At 1,000-token prefill and decode step 50, it retained all
1,049 KV positions, reached 0.999835 cosine, and matched the top-5 set; the
order of two tied entries differed.

Gemma2 is really slow rn ik, its a older and a kind of bad arch for our case

## Every public profile on the newly added architectures

| Model | Profile | Thin tok/s | Peak GiB | Min cosine | Top-1 | Top-5 set | Top-5 order |
|:---|:---|---:|---:|---:|:---:|:---:|:---:|
| Gemma-2-2B-IT | safe | 50.10 | 4.903 | 0.999936 | yes | yes | yes |
| Gemma-2-2B-IT | balanced | 49.41 | 4.903 | 0.999936 | yes | yes | yes |
| Gemma-2-2B-IT | max-performance | 57.29 | 5.463 | 0.999555 | yes | yes | yes |
| Gemma-2-2B-IT | lab | 50.38 | 4.903 | 0.999936 | yes | yes | yes |
| Gemma-2-2B-IT | Transformers | 53.58 | 4.901 | 1.000000 | yes | yes | yes |
| TinyLlama-1.1B-Chat | safe | 123.27 | 2.064 | 0.999836 | yes | no | no |
| TinyLlama-1.1B-Chat | balanced | 131.62 | 2.064 | 0.999845 | yes | no | no |
| TinyLlama-1.1B-Chat | max-performance | 180.28 | 2.137 | 0.997788 | yes | no | no |
| TinyLlama-1.1B-Chat | lab | 107.81 | 2.064 | 0.999836 | yes | no | no |
| TinyLlama-1.1B-Chat | Transformers | 132.36 | 2.064 | 1.000000 | yes | yes | yes |
| Qwen3.5-0.8B | safe | 109.86 | 1.450 | 0.999811 | yes | no | no |
| Qwen3.5-0.8B | balanced | 114.49 | 1.450 | 0.999811 | yes | no | no |
| Qwen3.5-0.8B | max-performance | 115.91 | 1.453 | 0.999811 | yes | no | no |
| Qwen3.5-0.8B | lab | 105.12 | 1.450 | 0.999811 | yes | no | no |
| Qwen3.5-0.8B | max-max-perf | 145.08 | 1.444 | 0.997078 | yes | yes | no |
| Qwen3.5-0.8B | Transformers | 64.33 | 1.452 | 1.000000 | yes | yes | yes |
| Gemma-4-E2B | safe | 35.05 | 6.490 | n/a | n/a | n/a | n/a |
| Gemma-4-E2B | balanced | 34.26 | 6.490 | n/a | n/a | n/a | n/a |
| Gemma-4-E2B | max-performance | 34.24 | 6.490 | 0.999090 | yes | yes | no |
| Gemma-4-E2B | lab | 35.02 | 6.490 | n/a | n/a | n/a | n/a |
| Gemma-4-E2B | max-max-perf | 43.60 | 6.49 | 0.999671 | yes | yes | no |
| Gemma-4-E2B | Transformers | 1.62 | 6.620 | 1.000000 | yes | yes | yes |
| OLMoE-1B-7B-0924-Instruct | safe | 8.99 | 6.935 | n/a | n/a | n/a | n/a |
| OLMoE-1B-7B-0924-Instruct | balanced | 30.65 | 6.832 | n/a | n/a | n/a | n/a |
| OLMoE-1B-7B-0924-Instruct | max-performance | 31.28 | 6.772 | 0.998586 | yes | yes | no |
| OLMoE-1B-7B-0924-Instruct | lab | 10.23 | 6.935 | n/a | n/a | n/a | n/a |
| OLMoE-1B-7B-0924-Instruct | max-max-perf | 33.70 | 6.82 | 0.999218 | yes | yes | no |
| OLMoE-1B-7B-0924-Instruct | Transformers | 9.22 | 6.450 | 1.000000 | yes | yes | yes |

Quality rows are the minimum across the public quick suite at prefill lengths
1 and 128 and decode steps 1 and 10. Gemma max-performance was additionally
checked after its adaptive INT8 switch at decode step 50; that lower value is
reported above. The HF correctness reference runs on the requested CUDA device
and is unloaded before ThinTensor starts.

## Reproduce

```bash
thintensor validate MODEL.thin \
  --hf-model ./original-model \
  --profile max-performance \
  --suite quick --require-tier ranking

thintensor bench MODEL.thin \
  --profiles safe,balanced,max-performance,lab \
  --warmup 10 --steps 200 \
  --hf-model ./original-model --json
```

Short runs and isolated microkernels are not accepted as end-to-end wins.
Results are architecture-, model-, context-, driver-, clock-, and
power-dependent.

## Benchmark Methodology and Metrics Calculation

To ensure accurate, robust, and reproducible results, the benchmarking and correctness scoring flow is executed as follows:

### 1. Benchmark Execution Workflow
Benchmarks are run in isolated, dedicated Python processes for each engine (ThinTensor vs. Hugging Face Transformers) to prevent memory leakage or process interference:
- **ThinTensor Baseline**:
  - The model is loaded via the `thin_runtime.py` engine.
  - **Weight Fit Planning**: The runtime runs `_automatic_fit_plan` to estimate weight size. If the model's weights do not fit within the available GPU VRAM (accounting for system/display manager overhead), it dynamically configures streaming weight residency (`--weight-residency stream`) and streams layers/experts block-by-block from CPU/RAM to the GPU. If the weights fit, it uses fully resident weights (`--weight-residency all`).
  - **Timing**: Prefills the prompt (default "Hello"), decodes a number of warmup tokens (default 10) to initialize caches and compile/warm up Triton kernels, and then times the next decode steps (default 200) using wall-clock time (`time.perf_counter()`).
- **Transformers (Hugging Face) Baseline**:
  - The model is loaded via a dedicated script `bench_hf_transformers_decode.py` in BF16 precision.
  - **RAM/CPU Offloading**: For larger models (e.g. Gemma-4-E2B, OLMoE), loading the model fully on memory-constrained GPUs (like the RTX 5050 Laptop GPU with 7.57 GiB VRAM) causes CUDA OOM. To prevent OOM, RAM offloading is enabled via `device_map="auto"`. Weights that exceed VRAM capacity are kept in CPU system RAM and loaded/swapped as needed.
  - **Timing**: Like ThinTensor, it pre-runs a prompt prefill, decodes warmup tokens (default 10), and then times `steps` (default 200) decode steps. The decode loop avoids GPU-to-CPU synchronization (e.g. calling `.item()` inside the timed loop) to measure pure GPU decoding speed.

### 2. Metrics & Scores Calculation
- **Thin tok/s & HF tok/s (Tokens per Second)**: Calculated as:
  $$\text{Tokens/s} = \frac{\text{Measured Steps}}{\text{Decode Elapsed Time (seconds)}}$$
- **Speedup**: Calculated as:
  $$\text{Speedup} = \frac{\text{Thin tok/s}}{\text{HF tok/s}}$$
- **Thin/HF Peak GiB (Peak GPU Memory)**: Tracks peak allocated VRAM in GiB using `torch.cuda.max_memory_allocated()`. This measures active tensors allocated by PyTorch's allocator during the decode phase (excluding unallocated cache reserve or general display server memory).
- **VRAM Delta**: Calculated as:
  $$\text{VRAM Delta} = \frac{\text{Thin Peak VRAM} - \text{HF Peak VRAM}}{\text{HF Peak VRAM}} \times 100\%$$
- **Min Cosine (Cosine Similarity)**: In `compare_hf_thin_logits.py`, the logits of the final linear projection layer are collected for both models. Cosine similarity is calculated for the target token index at each step. The minimum cosine similarity found across all decode steps (e.g., step 1 and step 10) and prefill lengths (e.g., 1 and 128) is reported.
- **Top-1 Match**: A boolean check indicating whether the token ID with the highest probability (highest logit value) is identical between ThinTensor and HF at all decode steps.
- **Top-5 Set**: A boolean check indicating whether the set of top-5 highest-probability token IDs is identical between the two models at all steps (ignoring their internal sorted order).
- **Top-5 Order**: A boolean check indicating whether the exact sorted order of the top-5 token IDs matches between ThinTensor and HF at all steps.
- **KV Retention**: Specifies the KV cache precision format (e.g. `full BF16`).
