# Full-logit llama.cpp Q8_0 comparison

All modes use identical HF token IDs and HF BF16 teacher trajectories.

| Mode | Min centered-logit cosine | Min raw-logit cosine | Top1 all pass | Min top5 | Max JS distance | Max TV |
|:---|---:|---:|:---:|---:|---:|---:|
| thin_bf16_source | 1.000000 | 1.000000 | True | 1.000 | 0.000000 | 0.000000 |
| Q4_K_b9986 | 0.997281 | - | True | 1.000 | 0.052306 | 0.052798 |

llama.cpp returns complete log-probabilities, not raw logits. Centered-logit cosine removes the unknown additive log-softmax constant and is therefore the direct cross-runtime cosine.
