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
| Gemma-4-E2B | candidate | 200 | 35.05 | OOM | n/a | 6.490 / OOM | n/a | n/a | n/a | n/a | n/a | full BF16 |
| OLMoE-1B-7B-0924-Instruct | candidate | 200 | 31.28 | OOM | n/a | 6.772 / OOM | n/a | n/a | n/a | n/a | n/a | full BF16 |

`candidate` is deliberate for TinyLlama: the short suite retained exact top-1
and 0.997788 minimum cosine, but one BF16 fifth-place cutoff tie changed the
strict top-5 set. At 1,000-token prefill and decode step 50, it retained all
1,049 KV positions, reached 0.999835 cosine, and matched the top-5 set; the
order of two tied entries differed.

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
| Qwen3.5-0.8B | Transformers | 64.33 | 1.452 | 1.000000 | yes | yes | yes |
| Gemma-4-E2B | safe | 35.05 | 6.490 | n/a | n/a | n/a | n/a |
| Gemma-4-E2B | balanced | 34.26 | 6.490 | n/a | n/a | n/a | n/a |
| Gemma-4-E2B | max-performance | 34.24 | 6.490 | n/a | n/a | n/a | n/a |
| Gemma-4-E2B | lab | 35.02 | 6.490 | n/a | n/a | n/a | n/a |
| Gemma-4-E2B | Transformers | OOM | OOM | n/a | n/a | n/a | n/a |
| OLMoE-1B-7B-0924-Instruct | safe | 8.99 | 6.935 | n/a | n/a | n/a | n/a |
| OLMoE-1B-7B-0924-Instruct | balanced | 30.65 | 6.832 | n/a | n/a | n/a | n/a |
| OLMoE-1B-7B-0924-Instruct | max-performance | 31.28 | 6.772 | n/a | n/a | n/a | n/a |
| OLMoE-1B-7B-0924-Instruct | lab | 10.23 | 6.935 | n/a | n/a | n/a | n/a |
| OLMoE-1B-7B-0924-Instruct | Transformers | OOM | OOM | n/a | n/a | n/a | n/a |

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
