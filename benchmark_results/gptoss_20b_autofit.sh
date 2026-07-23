#!/bin/bash
set -e

THINTENSOR_PINNED_STAGING=0 python scripts/thin_runtime.py run \
    /home/satvik/.cache/thintensor/archives/openai--gpt-oss-20b.thin \
    --device cuda \
    --dtype bf16 \
    --steps 200 \
    --warmup-steps 10 \
    --attention-mode causal_kv \
    --kernel-backend triton \
    --attention-backend triton_fused \
    --fused-residual-norm \
    --fused-rope \
    --embed-fp8 \
    --lm-head-fp8 \
    --gpu-weight-budget 7082336256 \
    --json > benchmark_results/gptoss_20b_autofit.json
