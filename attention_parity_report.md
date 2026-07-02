# Attention parity report

## Selected-layer component audit

- Layer: `31`
- Token position: `136`
- First failing component: `attention_probs`
- Interpretation: attention inputs, RoPE, scores, probabilities, and value mixing are compared independently from later projection drift.
- A failure first appearing at final norm/logits after earlier attention components pass is accumulated projection/quantization drift, not a causal-KV semantic mismatch.

| component | HF shape | Thin shape | HF dtype | Thin dtype | max abs | mean abs | cosine | pass |
|:---|:---|:---|:---|:---|---:|---:|---:|:---:|
| layer_input | [2048] | [2048] | bfloat16 | bfloat16 | 0.031250 | 0.005465 | 0.999819 | True |
| post_input_rmsnorm | [2048] | [2048] | bfloat16 | bfloat16 | 0.093750 | 0.007432 | 0.999897 | True |
| q_projection | [2048] | [2048] | bfloat16 | bfloat16 | 0.062500 | 0.014548 | 0.999930 | True |
| k_projection | [512] | [512] | bfloat16 | bfloat16 | 0.062500 | 0.015108 | 0.999956 | True |
| v_projection | [512] | [512] | bfloat16 | bfloat16 | 0.029297 | 0.005494 | 0.999794 | True |
| q_after_q_norm | [2048] | [2048] | bfloat16 | bfloat16 | 0.062500 | 0.014548 | 0.999930 | True |
| k_after_k_norm | [512] | [512] | bfloat16 | bfloat16 | 0.062500 | 0.015108 | 0.999956 | True |
| q_after_rope | [2048] | [2048] | bfloat16 | bfloat16 | 0.062500 | 0.014548 | 0.999930 | True |
| k_after_rope | [512] | [512] | bfloat16 | bfloat16 | 0.062500 | 0.015108 | 0.999956 | True |
| attention_scores | [16, 137] | [4, 4, 137] | bfloat16 | bfloat16 | 0.390625 | 0.033888 | 0.999896 | True |
| attention_probs | [16, 137] | [4, 4, 137] | bfloat16 | bfloat16 | 0.006348 | 0.000117 | 0.998996 | False |
| attention_output_before_o_proj | [2048] | [2048] | bfloat16 | bfloat16 | 0.012695 | 0.001621 | 0.997890 | False |
| o_proj_output | [2048] | [2048] | bfloat16 | bfloat16 | 0.016846 | 0.000892 | 0.997857 | False |
| post_attention_residual | [2048] | [2048] | bfloat16 | bfloat16 | 0.046875 | 0.005576 | 0.999811 | True |
| post_attention_rmsnorm | [2048] | [2048] | bfloat16 | bfloat16 | 0.035156 | 0.006770 | 0.999752 | True |
| gate_projection | [11008] | [11008] | bfloat16 | bfloat16 | 0.062500 | 0.005639 | 0.999850 | True |
| up_projection | [11008] | [11008] | bfloat16 | bfloat16 | 0.026611 | 0.005134 | 0.999761 | True |
| gated_activation | [11008] | [11008] | bfloat16 | bfloat16 | 0.017578 | 0.000984 | 0.999639 | True |
| down_projection | [2048] | [2048] | bfloat16 | bfloat16 | 0.012238 | 0.002017 | 0.999677 | True |
| final_residual_after_mlp | [2048] | [2048] | bfloat16 | bfloat16 | 0.046875 | 0.006095 | 0.999803 | True |
| final_norm | [2048] | [2048] | bfloat16 | bfloat16 | 0.203125 | 0.013124 | 0.999812 | True |
| final_logits | [128256] | [128256] | bfloat16 | bfloat16 | 0.500000 | 0.081550 | 0.999823 | True |
