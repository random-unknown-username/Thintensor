# Decode bottleneck profiles

| profile | tok/s | steady tok/s | ms/token | qkv | qk | softmax | value mix | o proj | gate | up | down | lm head | argmax | launches | resident weights | KV allocated | temp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bf16 | 33.566 | 33.542 | 29.792 | 0.055 | 0.017 | 0.015 | 0.024 | 0.038 | 0.306 | 0.000 | 0.168 | 1.722 | 0.041 | 434 | 6150197248 | 37748736 | 619020 |
| quality | 47.515 | 47.496 | 21.046 | 0.042 | 0.034 | 0.024 | 0.030 | 0.047 | 0.081 | 0.069 | 0.102 | 1.547 | 0.041 | 434 | 4076113920 | 37748736 | 619020 |

## Ranked bottlenecks

| Component | BF16 ms/token | Quality FP8 ms/token | BF16 % | Quality % | Read estimate BF16 / quality | Launches BF16 / quality | Candidate optimizations |
|---|---:|---:|---:|---:|---:|---:|---|
| Gate/up projections | 11.009 | 5.428 | 39.7% | 24.1% | 3246391296 / 1626365952 | 36 / 72 | persistent FP8 GEMV, role-specific row scheduling |
| Down projection | 6.031 | 3.674 | 21.7% | 16.3% | 1623195648 / 1172471808 | 36 / 36 | loop-tile tuning, split-K only if end-to-end wins |
| Attention core | 1.999 | 3.169 | 7.2% | 14.1% | 1024000 / 1024000 | 108 / 108 | context-aware block-split attention at long context |
| QKV projections | 1.981 | 1.528 | 7.1% | 6.8% | 452984832 / 452984832 | 36 / 36 | net-zero contiguous packing, persistent grouped GEMV |
| LM head + argmax | 1.763 | 1.588 | 6.4% | 7.0% | 525336576 / 525336576 | 2 / 2 | guarded shortlist, persistent vocab reduction, prepacked head |
| O projection | 1.352 | 1.695 | 4.9% | 7.5% | 301989888 / 301989888 | 36 / 36 | role-specific GEMV, exact layout packing |

Component timings come from an event-instrumented diagnostic pass after the real timed decode loop. Percentages rank work within that diagnostic pass; only steady full-decode tok/s decides retention.
