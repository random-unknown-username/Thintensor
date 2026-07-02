# Model schema support

ThinTensor discovers decoder semantics from HF config fields and tensor roles.
Model names are labels, not execution switches.

## Executable schemas

| Schema trait | Conversion | Native runtime | Validation |
|:---|:---:|:---:|:---|
| Separate Q/K/V + gated dense MLP | yes | optimized Triton/BF16/selective FP8 | real SmolLM3-3B |
| Fused QKV + separate gate/up | yes | reference GPU path | synthetic archive |
| Separate Q/K/V + fused gate/up | yes | reference GPU path | synthetic archive |
| Fused QKV + fused gate/up | yes | reference GPU path | synthetic archive |
| MHA, GQA, MQA | yes | yes | real GQA plus synthetic geometry |
| Per-layer full/sliding attention | yes | yes | synthetic causal-KV window test |
| Learned attention-sink logits | yes | yes | synthetic reference comparison |
| Default, linear, dynamic, YaRN, LongRoPE, Llama-3 RoPE | metadata | Transformers-standard initializer | default real test; YaRN synthetic smoke |
| Packed top-k sparse MoE | yes | GPU-only router and selected experts | synthetic independent reference |
| Packed MXFP4 sparse MoE | yes | direct packed Triton matvec | realistic kernel microbench and synthetic stream |
| GPTQ-style packed integer roles | yes, storage preserved | codec adapter required | synthetic Q4 convert/verify/role-plan |
| Biases on Q/K/V/O/router/experts | yes | yes | synthetic MoE |
| Tied or untied LM head | yes | yes | real models |

## Planned but not yet native

- Separate-per-expert MoE checkpoints such as un-repacked Mixtral layouts:
  role discovery works, but streaming requires archive-time expert stacking to
  avoid synchronizing router IDs to the CPU.
- GPT-style learned absolute position embeddings and Conv1D transposed weights.
- ALiBi and other non-RoPE position biases.
- Encoder-decoder, multimodal, state-space, recurrent, and custom remote-code
  operators.
- Native GPTQ/AWQ/bitsandbytes Q8/Q4/Q2 kernels. Their source metadata is
  preserved and lossy remapping is blocked, but they currently require an
  explicit codec adapter.
- Gemma 2/3/4 residual/norm/soft-cap semantics. These checkpoints must not be
  treated as plain Llama merely because their projection names match.

Use:

```bash
python3 scripts/inspect_hf_compat.py --hf-model /path/to/model --json
```

The report distinguishes:

1. whether config/tensor roles compile into an execution plan;
2. whether the current runtime has every required operator;
3. whether source quantization has a native kernel or a fallback;
4. which optimizations are eligible for real-decode autotuning.

A schema match is not a performance claim. The automatic optimizer only retains
a configuration after real same-model decode improves by at least 2%, and a
30% claim requires a measured 1.30x speedup over the same-precision baseline.
