# Safe claims

- ThinTensor now discovers common causal-decoder schemas from config and tensor roles rather than rejecting unknown model-type labels. Separate/fused dense projections and packed sparse-MoE archives pass synthetic convert/verify/runtime integration tests.
- The direct packed MXFP4 selected-expert kernel is numerically aligned with explicit dequantization in the tested shapes. It measured 0.634 ms versus 18.183 ms for top-4 5760x2880 gate/up and 0.292 ms versus 8.422 ms for top-4 2880x2880 down. These are kernel microbenchmarks, not full-model rates.
- New conversions physically order pages by the execution tape and preserve per-page quant scheme, bits, group size, scale-page, and backend-layout metadata.
- Generalization did not regress the local SmolLM path: BF16 measured 32.38 tok/s over 200 tokens and retained `exact_pass`; quality-first selective FP8 measured 48.03 tok/s over 300 tokens and retained minimum required-test cosine 0.99804.
- SmolLM3-3B exact BF16 mode passes the strict full-recompute eager-HF gate: all required top-1/top-5 and greedy generations match, minimum logit cosine is 0.999316, `attention_equivalent=true`, and `hf_equivalent=true`.
- SmolLM3-3B causal attention is structurally aligned with eager HF for the audited layer/token: Q/K/V, selective RoPE, attention scores, softmax probabilities, and value mixing pass the component checks.
- Optimized BF16 is the safe default. The 200-token causal benchmark measured 31.44 tok/s; the correctness run preserved all tested top-1/top-5 results and reached the `ranking_pass` tier.
- The retained quality-first selective-FP8 mode quantizes gate/up in every layer and down projection in layers 8:28. A 500-token run measured 46.41 tok/s, saved 2,070,749,184 resident weight bytes, preserved all required top-1/top-5 results, and had worst required-test cosine 0.99804.
- In the broader cached-HF stress subset, the quality-first mode's worst checkpoint cosine was 0.99379. The faster mode that also quantizes O layers 4:32 fell to 0.98471 on the multi-turn chat case, so O FP8 is not part of the quality-first recommendation.
- Full causal-KV stress completed at 512- and 1024-token prompts without OOM. Both BF16 exact and the tested selective-FP8 mode matched all 16 greedy continuation tokens; KV storage grew from 38,928,384 to 76,677,120 bytes as context doubled.
- `--mlp-fp8` is selective quantization, not BF16. With the exact BF16 LM head and the RTX 5050 real-decode-validated row-block m2 backend, it measured 50.61 tok/s and preserved all tested top-1/top-5 and greedy tokens. Its worst tested cosine was 0.98378, so it is ranking-equivalent but not HF-equivalent.
- `--mlp-fp8 --o-proj-fp8` is an experimental selective-quantization mode. It measured 51.53 tok/s and saved 2,582,028,288 resident weight bytes while preserving all tested top-1/top-5 and greedy tokens; its worst cosine was 0.97096.
- Keeping `o_proj` BF16 reduces the measured worst-case logit drift from cosine 0.97096 to 0.98378 while retaining a measured 50+ tok/s configuration.
- The guarded LM-head profile uses FP8 only to shortlist 64 vocabulary IDs and
  recomputes final candidate logits from the resident tied BF16 head. Balanced
  500-token medians improved full BF16 from 32.440 to 33.718 tok/s (+3.94%)
  and quality FP8 from 47.608 to 49.798 tok/s (+4.60%).
- Guarded LM-head public logits, required correctness, and 13-case stress
  metrics matched each path's baseline. It adds 263,181,312 resident bytes and
  is an explicit model-validated profile, not a universal raw-Head8 default.
- The guarded speedup is attributable to estimated head-read reduction:
  261,893,120 fewer bytes/token. Launch-count reduction is not used as evidence
  for this claim.
- Final current-source standalone 500-token checks measured 35.206 tok/s for
  guarded BF16 and 50.884 tok/s for guarded quality, versus 34.310 and 48.625
  tok/s for their unguarded paths. Balanced ABBA remains the acceptance
  evidence because laptop power and thermals move absolute standalone rates.
- On top of the guarded quality profile, widening down-projection FP8 from
  layers 8:28 to 4:32 and enabling O-proj FP8 for 4:32 improved balanced
  500-token decode from 48.811 to 51.376 tok/s (+5.26%). Estimated weight reads
  fell by 297,795,584 bytes/token, and resident weights fell by the same amount.
- That MLP/O profile preserved every required and 13-case stress checkpoint
  top-1, required minimum cosine 0.998410, stress minimum top-5 0.800, and
  exact 16-token continuations at both 512- and 1024-token contexts. It remains
  opt-in because its worst stress cosine was 0.988403.
- Exact BF16 CPU weight offload now completes at 1/2/4/6 GB GPU budgets.
  The full HF matrix produced identical correctness metrics at every budget:
  zero top-1 failures, complete top-5 overlap, minimum cosine 0.9973128438,
  and minimum generated-token match 0.9.
- The 6 GB short capacity run retained 5,681,459,200 GPU weight bytes and
  transferred 726,986,193 bytes/token, versus 681,586,688 resident bytes and
  5,499,591,773 H2D bytes/token at 1 GB. These are exact-residency capacity
  measurements, not speedup claims.
## General schema and quantization claims

- Model labels no longer select the runtime. HF config fields plus tensor roles
  compile dense, fused-projection, packed-MoE, attention-window, sink, RoPE,
  bias, tied-head, residency, and source-quantization plans.
- Source Q8/Q4/Q2 metadata is preserved. Integer Q4 is not called FP4, and
  lower-bit requantization is never selected implicitly.
- Packed MXFP4 selected-expert matvec is native and validated at kernel and
  synthetic-runtime level. GPTQ/AWQ-style integer packing remains an explicit
  native-executor gap.
