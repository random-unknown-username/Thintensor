# ThinTensor schema compatibility report

## Result

ThinTensor no longer selects execution from a fixed model-type allowlist.
Conversion compiles HF config plus tensor names into semantic roles and emits
those traits in the `.thin` manifest.

## Current evidence

| Fixture/model | Schema | Convert/verify | Runtime evidence |
|:---|:---|:---:|:---|
| SmolLM3-3B | separate QKV + gated dense GQA | existing real archive | BF16 32.38 tok/s; exact pass |
| SmolLM3-3B quality FP8 | same schema, role-selective precision | existing real archive | 48.03 tok/s over 300 tokens; min cosine 0.99804 |
| Synthetic fused dense | fused QKV + fused gate/up | pass | 10-token causal-KV GPU smoke |
| Synthetic packed MoE | biased packed top-k experts | pass | independent-reference cosine above 0.99999 |
| Synthetic MXFP4 MoE | packed top-k experts, sinks, sliding KV | pass | native packed CUDA and 64 MB stream/offload smoke |
| Synthetic GPTQ Q4 dense | separate QKV + gated dense | pass; packed pages/scales/zeros/group index retained | planner reports native codec gap instead of expanding silently |
| Public GPT-OSS-20B metadata | packed MXFP4 MoE, alternating sliding/full GQA, YaRN, sinks | role plan compiles | actual weights not run |

The synthetic MXFP4 stream kept the declared two-token window, evicted obsolete
KV blocks, and completed with GPU-only router selection. It validates operator
integration, not large-model speed.

## Common schema coverage

- separate or fused QKV;
- separate or fused gated MLP;
- MHA, GQA, and MQA geometry;
- packed top-k sparse MoE;
- default and standard Transformers RoPE variants;
- per-layer sliding/full attention;
- learned attention sinks;
- projection/router/expert biases;
- tied and untied output heads;
- all-resident and stream/CPU-offload residency;
- BF16, scaled selective FP8, and packed MXFP4 execution.

## Remaining gaps

- archive-time packing of separate-per-expert MoE;
- native adapters for GPTQ/AWQ/bitsandbytes/compressed-tensors Q8/Q4/Q2;
- Gemma-family embedding scale, one-centered RMSNorm, soft caps, and
  four-norm residual blocks;
- GPT-style learned absolute positions and transposed Conv1D;
- ALiBi;
- encoder-decoder, multimodal, state-space, and custom operators.

Compatibility does not imply a 30% speedup. The generalized optimizer derives
candidates from the compiled schema and only retains measured real-decode wins.
