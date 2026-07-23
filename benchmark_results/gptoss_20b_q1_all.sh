#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python3 scripts/thin_runtime.py run \
    --device cuda \
    --dtype bf16 \
    --steps 200 \
    --warmup-steps 10 \
    --attention-mode causal_kv \
    --kernel-backend triton \
    --attention-backend triton_fused \
    --mlp-fp8 \
    --gate-up-fp8 \
    --down-proj-fp8 \
    --qkv-fp8 \
    --attn-proj-fp8 \
    --o-proj-fp8 \
    --residency stream \
    --prefetch 0 \
    --kv-policy local \
    --pinned-staging \
    --cuda-graphs \
    --packed-expert-q1-layers "all" \
    --fused-mlp \
    --fused-residual-norm \
    --fused-rope \
    --lm-head-fp8 \
    --embed-fp8 \
    --fp8-layers all \
    --down-fp8-layers all \
    --qkv-fp8-layers all \
    --o-fp8-layers all \
    --fused-scaled-mlp \
    --experimental-int8-tensorcore \
    --fp8-residual-terms 0 \
    --fp8-residual-layers all \
    --lm-head-backend triton \
    --lm-head-argmax-mode triton_persistent \
    --gpu-weight-budget 7082336256 \
    --json \
    /home/satvik/.cache/thintensor/archives/openai--gpt-oss-20b.thin > benchmark_results/gptoss_20b_q1_all.json
