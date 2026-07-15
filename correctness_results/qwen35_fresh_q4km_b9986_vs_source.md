# Full-logit llama.cpp Q8_0 comparison

All modes use identical HF token IDs and HF BF16 teacher trajectories.

| Mode | Min centered-logit cosine | Min raw-logit cosine | Top1 all pass | Min top5 | Max JS distance | Max TV |
|:---|---:|---:|:---:|---:|---:|---:|
| thin_bf16_source | 1.000000 | 1.000000 | True | 1.000 | 0.000000 | 0.000000 |
| llamacpp_q4_k_m_b9986 | 0.894908 | - | True | 0.400 | 0.057975 | 0.062302 |

llama.cpp returns complete log-probabilities, not raw logits. Centered-logit cosine removes the unknown additive log-softmax constant and is therefore the direct cross-runtime cosine.
