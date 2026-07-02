# Unsafe claims

- Do not claim that every Hugging Face model is supported or 30% faster. Current native scope is common causal-decoder schemas; separate-expert MoE, GPT-style absolute positions/Conv1D, ALiBi, encoder-decoder, multimodal, and custom operators still need adapters.
- Do not claim GPT-OSS-20B end-to-end performance. Its public config/index compile into the packed-MoE/MXFP4 plan, but the actual 20B weights were not downloaded or benchmarked.
- Do not present the 28x MXFP4 result as model speedup. It compares one packed selected-expert kernel against expand-then-matvec.
- Do not silently convert Q8 to FP8/FP4, integer Q4 to FP4, or Q4 to Q2. These are different encodings and any cross-format/lower-bit conversion is opt-in and correctness-gated.
- Do not call any body-FP8 result BF16 or fully HF-equivalent.
- Do not set `hf_equivalent=true` for MLP/O FP8: it is `ranking_pass` and `quantized_experimental`, with cosine drift below the strict exact threshold.
- Do not call Head8 fully ranking-equivalent for the required 1/8/128 by 1/10 run: one tested top-1 differed, so its tier is `experimental`.
- Do not use the cached-HF stress matrix as the strict equivalence gate. Cached HF and full-recompute eager HF are numerically different paths; strict `exact_pass` is established by the full-recompute eager comparison.
- Do not claim that the retained selective-FP8 mode is BF16 or fully HF-equivalent. Its required-test cosine is 0.99650 and it remains opt-in.
- Do not present short teacher-forced token agreement as free-running generation equivalence. Free-running results must be reported separately.
- Do not claim stochastic sampled sequences are equivalent. In the three fixed-seed policies, small distribution changes caused early free-running divergence even for BF16; the retained FP8 configurations had checkpoint total variation as high as roughly 0.08-0.09.
- Do not call the 49.88 tok/s O-FP8 configuration the quality-preserving default. Its broader multi-turn checkpoint cosine fell to 0.98471.
- Do not use Qwen3-0.6B `current_only_smoke` speed as an HF-equivalent result. It intentionally skips historical attention and reports `not_hf_equivalent=true`.
- Do not claim the prior 240+ Qwen Head8 rate for causal decode. The final 100-step causal-KV measurement was 97.93 tok/s; the remaining gap is causal-attention overhead.
- Do not claim Qwen3-4B was revalidated in this run; no local `Qwen3-4B.thin` archive was available.
## Generalization limits

- Do not claim that arbitrary Hugging Face models execute natively. The current
  optimized executor targets common RoPE decoder schemas; GPT Conv1D/absolute
  positions, ALiBi, Gemma 2+ residual semantics, separate-expert streaming,
  encoder-decoder, multimodal, state-space, recurrent, and custom remote-code
  operators still require adapters.
- Do not claim a universal 30% speedup. Compatibility and performance are
  separate gates, and speedups are retained only from same-model real decode.
- Do not claim native GPTQ/AWQ/Q8/Q4/Q2 execution from metadata preservation.
- Do not claim a microbenchmark win as a decode win. Retention requires warmed
  200/500-token full decode.
- Do not rank large projection kernels from repeated use of one layer's
  weights. A 22.5 MB projection fits in the 32 MB L2 on the tested GPU, while
  the real quality path streams roughly 4 GB/token; row scheduling measured up
  to 1.88x in isolation but either failed full decode or correctness.
- Do not claim raw Head8 is safe unless the required and stress top-1/top-5
  gates pass. The guarded shortlist profile is distinct because final
  candidate logits are recomputed in BF16.
- Do not claim CPU offload is faster unless the full decode measurement shows
  it; exact CPU residency is primarily a memory-capacity option here.
- Do not present the 10-token 1/2/4/6 GB weight-residency curve as warmed
  500-token throughput. It is a same-command capacity comparison, and every
  streamed tier is slower than all-resident execution.
- Do not equate the page pool's active resident-weight budget with CUDA
  allocator reservation. Peak allocated stayed near each budget, but the
  caching allocator reserved up to about 6.77 GB while recycling streamed
  page shapes; both numbers must be reported.
- Do not call KV compression lossless unless it stores the original BF16 values
  exactly. Reduced GPU residency through exact CPU pages is not quantization.
- Do not claim launch fusion helps from launch counts alone. A fusion must
  improve warmed 200/500-token decode; fused residual+RMSNorm is the negative
  control.
- Do not compare top-k-only llama.cpp or Ollama probabilities as if they were
  full-vocabulary cosine measurements.
- Do not claim the guarded 64-token shortlist is universally safe across
  models. The retained result is validated for this SmolLM3-3B profile and
  carries an extra 263,181,312 resident bytes.
- Do not call down 4:32 plus O-proj 4:32 BF16-equivalent or a universal
  default. It is selective FP8, its worst stress cosine was 0.988403, and the
  retained claim is specific to the tested SmolLM3-3B profile.
- Do not claim QKV FP8 as retained. Wider QKV ranges changed a stress
  checkpoint top-1; paired QKV+O 12:24 restored that checkpoint but measured
  only +0.05% versus the simpler O-only profile.

## Rejected or experimental hot-path candidates

- Two-stage BF16 LM-head reduction lost in real decode: 47.14 tok/s versus
  47.79 tok/s for the retained path.
- Fused single-token attention matched ThinTensor reference logits at roughly
  0.99998 cosine but produced no real decode gain, so it is opt-in.
- Quality-body FP8 plus Head8 reached 50.44 tok/s, but changed top-1 at prefill
  128 / step 10 (cosine 0.99691, generation match 0.8). It is experimental.
