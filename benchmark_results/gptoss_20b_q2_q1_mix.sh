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
    --mlp-fp8 \
    --attn-proj-fp8 \
    --fp8-layers all \
    --down-fp8-layers all \
    --qkv-fp8-layers all \
    --o-fp8-layers all \
    --packed-expert-q2-layers 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19 \
    --packed-expert-q1-layers 20,21,22,23 \
    --fused-residual-norm \
    --fused-rope \
    --lm-head-fp8 \
    --embed-fp8 \
    --gpu-weight-budget 7082336256 \
    --cuda-graphs \
    --lm-head-argmax-mode triton_persistent \
    --json \
    --profile-runtime benchmark_results/q2_q1.prof > benchmark_results/gptoss_20b_q2_q1_mix.json
