# Operator-Graph Cross-Architecture Gate

Date: 2026-07-13

All fixtures are freshly initialized Transformers models saved as BF16
safetensors, converted through the public `thintensor convert` command, and
executed as exact BF16 ThinTensor archives. Correctness compares the complete
128-token vocabulary on causal teacher-forced trajectories. Speed uses 10
warmup plus 200 measured decode tokens on CUDA; `safe` is used where it is the
fastest exact profile.

| contract | HF fixture | min cosine | min top-5 | all top-1 | exact attention/KV | Thin tok/s | HF tok/s | speedup |
|:--|:--|--:|--:|:--:|:--:|--:|--:|--:|
| separate-QKV dense | Llama, 2 layers | 0.999975 | 1.0 | yes | yes | 1158.27 | 1144.64 | 1.0119x |
| fused-QKV + fused gate/up dense | Phi-3, 2 layers | 0.999980 | 1.0 | yes | yes | 1311.89 | 1097.66 | 1.1952x |
| sliding-window GQA | Mistral, window 4 | 0.999981 | 1.0 | yes | yes | 1311.52 | 929.96 | 1.4103x |
| separate-expert MoE | Mixtral, 4 experts/top-2 | 0.999986 | 1.0 | yes | yes | 956.49 | 656.22 | 1.4576x |
| packed-expert + sliding/full hybrid | GPT-OSS, 4 experts/top-2 | 0.999978 | 1.0 | yes | yes | 848.44 | 499.69 | 1.6979x |
| recurrent linear/full-attention hybrid | Qwen3.5, 2 layers | 0.999989 | 0.8 | yes | yes | 1246.35 | 730.95 | 1.7051x |

The common correctness command shape was:

```text
python scripts/compare_hf_thin_logits.py \
  --hf-model /tmp/thin-op-matrix-<fixture> \
  --archive /tmp/thin-op-matrix-<fixture>/model.thin \
  --device cuda --dtype bf16 --kernel-backend triton \
  --attention-backend torch --prefill-lens 1,4 --steps 4
```

The common speed command shape was:

```text
./thintensor bench /tmp/thin-op-matrix-<fixture>/model.thin \
  --profiles safe --steps 200 --warmup 10 --device cuda --dtype bf16 \
  --residency all --auto-quant off \
  --hf-model /tmp/thin-op-matrix-<fixture> --json
```

Fresh failures caught and fixed by this matrix:

- Runtime admission now compiles every layer into explicit normalization,
  sequence, and feed-forward blocks and admits heterogeneous dense/MoE graphs.
- Plain packed BF16 experts now honor GPT-OSS input-major
  `[expert, input, output]` storage for both gate/up and down projections.
- GPT-OSS descriptors restore the architecture's clipped interleaved SwiGLU
  limit (`7.0`) when older/config-minimal manifests omit it.
- Correctness accounting expects the effective last-layer attention window,
  instead of falsely labeling exact sliding-window retention as truncated KV.
- The HF speed runner retries with eager attention when an architecture such as
  GPT-OSS rejects SDPA, and records the implementation used.

Raw correctness and speed JSON for every row is retained under
`benchmark_results/operator_graph_matrix/`.
