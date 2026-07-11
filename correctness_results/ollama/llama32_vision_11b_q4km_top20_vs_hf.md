# Ollama Q4_K_M vs HF top-20 logprob comparison

- Scope: `top-20 only; Ollama 0.21 rejects top_logprobs > 20, so this is not full-vocabulary logit cosine`
- Ollama: `llama3.2-vision:11b` Q4_K_M
- HF: `/home/satvik/.cache/thintensor/models/unsloth--Llama-3.2-11B-Vision-Instruct`
- Min top-20 probability cosine: `0.720495`
- Min top-20 logprob cosine with floor: `0.505608`
- Min top-20 overlap: `0.550`
- Top-1 same all cases: `False`

| case | HF top1 | Ollama top1 | top1 same | top20 overlap | prob cosine | logprob cosine |
|---:|:---|:---|:---:|---:|---:|---:|
| 1 | `It` | `Hello` | False | 0.550 | 0.744020 | 0.530375 |
| 2 | `**` | `Here` | False | 0.700 | 0.720495 | 0.505608 |
| 3 | `Quant` | `Quant` | True | 0.600 | 0.988043 | 0.651239 |

Ollama API evidence: native `/api/chat` returned logprobs for `top_logprobs=20` but rejected `top_logprobs=100` and above. This artifact is therefore a top-20 comparison, not a full-vector cosine like ThinTensor/HF.
