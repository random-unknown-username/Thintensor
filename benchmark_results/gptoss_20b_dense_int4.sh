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
    --dense-int4 \
    --packed-expert-q2-layers all \
    --fused-residual-norm \
    --fused-rope \
    --lm-head-fp8 \
    --embed-fp8 \
    --gpu-weight-budget 7082336256 \
    --cuda-graphs \
    --lm-head-argmax-mode triton_persistent \
    --json > benchmark_results/gptoss_20b_dense_int4.json
