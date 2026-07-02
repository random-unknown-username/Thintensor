# Exact command log

Commands were run from `/home/satvik/Projects/thintensor-opus`.

## Correctness

```bash
python3 scripts/compare_hf_thin_logits.py --hf-model SmolLM3-3B --archive SmolLM3-3B.thin --device cuda --dtype bf16 --prompt Hello --prefill-lens 1,8,128 --steps 1,10 --exact-hf-mode --json --out hf_vs_thin_correctness_report.md > /tmp/smollm-bf16-exact-correctness.json
python3 scripts/compare_hf_thin_logits.py --hf-model SmolLM3-3B --archive SmolLM3-3B.thin --device cuda --dtype bf16 --prompt Hello --prefill-lens 1,8,128 --steps 1,10 --json --out /tmp/smollm-bf16-optimized-correctness.md > /tmp/smollm-bf16-optimized-correctness.json
python3 scripts/compare_hf_thin_logits.py --hf-model SmolLM3-3B --archive SmolLM3-3B.thin --device cuda --dtype bf16 --prompt Hello --prefill-lens 1,8,128 --steps 1,10 --lm-head-fp8 --json --out /tmp/smollm-head8-correctness.md > /tmp/smollm-head8-correctness.json
python3 scripts/compare_hf_thin_logits.py --hf-model SmolLM3-3B --archive SmolLM3-3B.thin --device cuda --dtype bf16 --prompt Hello --prefill-lens 1,8,128 --steps 1,10 --mlp-fp8 --o-proj-fp8 --json --out /tmp/smollm-mlp-o-correctness.md > /tmp/smollm-mlp-o-correctness.json
python3 scripts/compare_hf_thin_logits.py --hf-model SmolLM3-3B --archive SmolLM3-3B.thin --device cuda --dtype bf16 --prompt Hello --prefill-lens 1,8,128 --steps 1,10 --mlp-fp8 --json --out /tmp/smollm-mlp-only-correctness.md > /tmp/smollm-mlp-only-correctness.json
```

## Component debug

```bash
python3 scripts/compare_hf_thin_logits.py --hf-model SmolLM3-3B --archive SmolLM3-3B.thin --device cuda --dtype bf16 --prompt Hello --prefill-lens 1 --steps 1 --mlp-fp8 --o-proj-fp8 --compare-components --dump-layer-debug 0 --dump-token-debug 0 --out-debug-dir /tmp/thin_attention_debug --json --out /tmp/smollm-mlp-o-layer0-debug.md > /tmp/smollm-mlp-o-layer0-debug.json
```

## Benchmarks

```bash
python3 scripts/thin_runtime.py run SmolLM3-3B.thin --device cuda --dtype bf16 --steps 200 --residency all --kernel-backend triton-matvec --warmup-steps 10 --attention-mode causal_kv --json > /tmp/smollm-bf16-final-bench.json
python3 scripts/thin_runtime.py run SmolLM3-3B.thin --device cuda --dtype bf16 --steps 100 --residency all --kernel-backend triton-matvec --warmup-steps 10 --attention-mode causal_kv --mlp-fp8 --lm-head-backend row_block_m2 --json > /tmp/head-m2.json
python3 scripts/thin_runtime.py run SmolLM3-3B.thin --device cuda --dtype bf16 --steps 100 --residency all --kernel-backend triton-matvec --warmup-steps 10 --attention-mode causal_kv --mlp-fp8 --o-proj-fp8 --json > /tmp/smollm-mlp-o-final-bench.json
python3 scripts/thin_runtime.py run /tmp/thintensor-first-run/qwen3-0.6b.thin --device cuda --dtype bf16 --steps 100 --residency all --kernel-backend triton-matvec --warmup-steps 5 --attention-mode causal_kv --lm-head-fp8 --json > /tmp/qwen-head8-causal-final.json
```

## Archive and compile validation

```bash
cargo build --release
target/release/thintensor convert-hf SmolLM3-3B SmolLM3-3B.metadata.thin
target/release/thintensor verify SmolLM3-3B.thin
python3 -m py_compile thinruntime/gpu_runtime.py thinruntime/triton_kernels.py scripts/thin_runtime.py scripts/compare_hf_thin_logits.py
```

## 2026-07-01 parity, precision, and stress loop

```bash
python3 scripts/compare_hf_thin_logits.py --hf-model SmolLM3-3B --archive SmolLM3-3B.thin --device cuda --dtype bf16 --prompt Hello --prefill-lens 1,8,128 --steps 1,10 --exact-hf-mode --json --out hf_vs_thin_correctness_report.md

python3 scripts/thin_runtime.py run SmolLM3-3B.thin --device cuda --dtype bf16 --steps 500 --residency all --kernel-backend triton --warmup-steps 10 --attention-mode causal_kv --gate-up-fp8 --down-proj-fp8 --down-fp8-layers 8:28 --lm-head-backend triton --json

python3 scripts/compare_hf_thin_logits.py --hf-model SmolLM3-3B --archive SmolLM3-3B.thin --device cuda --dtype bf16 --prompt Hello --prefill-lens 1,8,128 --steps 1,10 --kernel-backend triton --gate-up-fp8 --down-proj-fp8 --down-fp8-layers 8:28 --json

python3 scripts/stress_validate_thin.py --hf-model SmolLM3-3B --archive SmolLM3-3B.thin --device cuda --dtype bf16 --modes bf16_exact,retained_fp8 --out stress_validation_report.md --json
```

Precision/backend grids and retained/rejected outputs are stored under
`benchmark_results/` and `correctness_results/`.

## Schema and quantization validation

```bash
python3 scripts/inspect_hf_compat.py --hf-model SmolLM3-3B --json
python3 scripts/inspect_hf_compat.py --hf-model /tmp/gpt-oss-schema --json
cargo run -- convert-hf /tmp/thin-moe-fixture /tmp/tiny-moe.thin
cargo run -- verify /tmp/tiny-moe.thin
cargo run -- convert-hf /tmp/thin-mxfp4-fixture /tmp/tiny-mxfp4.thin
cargo run -- verify /tmp/tiny-mxfp4.thin
python3 scripts/thin_runtime.py run /tmp/tiny-mxfp4.thin --device cuda --dtype bf16 --steps 10 --residency stream --cpu-offload --gpu-weight-budget 64mb --kernel-backend triton --warmup-steps 1 --attention-mode causal_kv --json
python3 scripts/bench_mxfp4_experts.py --experts 32 --top-k 4 --rows 5760 --cols 2880 --warmup 5 --repeats 20 --json
python3 scripts/bench_mxfp4_experts.py --experts 32 --top-k 4 --rows 2880 --cols 2880 --warmup 5 --repeats 20 --json
python3 scripts/auto_optimize.py --archive /tmp/tiny-mxfp4.thin --device cuda --steps 10 --warmup-steps 1 --max-rounds 2 --out-dir /tmp/auto_tiny_moe
```
## 2026-07-01 schema and hot-path generalization

```bash
cargo run --quiet -- convert-hf /tmp/thin-gptq-fixture /tmp/tiny-gptq.thin
cargo run --quiet -- verify /tmp/tiny-gptq.thin
python3 scripts/inspect_hf_compat.py --archive /tmp/tiny-gptq.thin --json
python3 scripts/bench_lm_head_modes.py SmolLM3-3B.thin --samples 40 --batch-size 10 --warmup 20 --out benchmark_results/smollm3_lm_head_modes.json
python3 scripts/thin_runtime.py run SmolLM3-3B.thin --device cuda --dtype bf16 --steps 250 --residency all --kernel-backend triton --warmup-steps 10 --attention-mode causal_kv --gate-up-fp8 --down-proj-fp8 --down-fp8-layers 8:28 --lm-head-backend triton --lm-head-argmax-mode torch --json
python3 scripts/thin_runtime.py run SmolLM3-3B.thin --device cuda --dtype bf16 --steps 250 --residency all --kernel-backend triton --warmup-steps 10 --attention-mode causal_kv --gate-up-fp8 --down-proj-fp8 --down-fp8-layers 8:28 --lm-head-backend triton --lm-head-argmax-mode triton_two_stage --json
python3 scripts/thin_runtime.py run SmolLM3-3B.thin --device cuda --dtype bf16 --steps 250 --residency all --kernel-backend triton --warmup-steps 10 --attention-mode causal_kv --attention-backend triton_fused --gate-up-fp8 --down-proj-fp8 --down-fp8-layers 8:28 --lm-head-backend triton --json
python3 scripts/thin_runtime.py run SmolLM3-3B.thin --device cuda --dtype bf16 --steps 300 --residency all --kernel-backend triton --warmup-steps 10 --attention-mode causal_kv --gate-up-fp8 --down-proj-fp8 --down-fp8-layers 8:28 --lm-head-fp8 --lm-head-backend triton --json
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 scripts/compare_hf_thin_logits.py --hf-model SmolLM3-3B --archive SmolLM3-3B.thin --device cuda --dtype bf16 --prompt Hello --prefill-lens 1,8,128 --steps 1,10 --kernel-backend triton --gate-up-fp8 --down-proj-fp8 --down-fp8-layers 8:28 --lm-head-fp8 --json --out correctness_results/quality_body_head8.md
```
