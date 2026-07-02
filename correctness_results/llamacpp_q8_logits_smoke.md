# Full-logit llama.cpp Q8_0 comparison

All modes use identical HF token IDs and HF BF16 teacher trajectories.

| Mode | Min centered-logit cosine | Min raw-logit cosine | Top1 all pass | Min top5 | Max JS distance | Max TV |
|:---|---:|---:|:---:|---:|---:|---:|
| hf_bf16 | 1.000000 | 1.000000 | True | 1.000 | 0.000000 | 0.000000 |
| thin_bf16 | 0.999596 | 0.998647 | True | 1.000 | 0.000205 | 0.000001 |
| thin_quality_8_28 | 0.995366 | 0.996249 | True | 0.800 | 0.000717 | 0.000009 |
| thin_quality_10_26 | 0.995235 | 0.995953 | True | 0.800 | 0.000614 | 0.000007 |
| llamacpp_q8_0 | 0.952666 | - | True | 0.800 | 0.027102 | 0.002155 |

llama.cpp returns complete log-probabilities, not raw logits. Centered-logit cosine removes the unknown additive log-softmax constant and is therefore the direct cross-runtime cosine.
