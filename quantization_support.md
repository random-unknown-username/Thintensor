# Quantization support

Storage encoding and compute precision are separate. ThinTensor never treats
Q4 integer, FP4, and MXFP4 as interchangeable merely because all use four bits.

| Source | Default policy | Native execution | Automatic lower-bit conversion |
|:---|:---|:---|:---|
| BF16/FP16 | preserve | BF16/Triton | none |
| Scaled FP8 | preserve scales | FP8 selected projections | none |
| MXFP4 packed experts | preserve blocks and E8M0 scales | direct selected-expert Triton matvec | none |
| Q8 / INT8 | preserve metadata; native adapter if available, otherwise dequantize | codec-dependent | never |
| Q4 / INT4 | preserve metadata; native adapter if available, otherwise dequantize | codec-dependent | never |
| Q2 / INT2 | preserve metadata; native adapter if available, otherwise dequantize | codec-dependent | never |
| GPTQ/AWQ/bitsandbytes/compressed-tensors | preserve method, bits, group, scales, and zero-point metadata | adapter required | never |

Consequences:

- Q8 to FP8 is a peer-format requantization, not a guaranteed compression or
  accuracy improvement.
- Q8 to Q4/FP4 and Q4 to Q2 are lossy opt-in experiments.
- Integer Q4 to FP4 is a cross-format conversion with different values/scales.
- A conversion is only selectable with `allow_requantize`, a matching kernel,
  and correctness validation.

The current MXFP4 kernel reads nibbles and per-32-value scales directly. On the
RTX 5050 Laptop GPU, a realistic top-4 `5760x2880` selected-expert microbench
measured 0.634 ms versus 18.18 ms for expand-then-matvec (28.7x kernel
speedup), with cosine approximately 1.0. This is not a full-model result.

A synthetic symmetric GPTQ Q4 archive with `qweight`, `scales`, `qzeros`, and
`g_idx` pages converts and verifies. Its execution plan compiles, but
`current_native_executor_ready` is false until a codec-specific kernel handles
GPTQ packing and zero-point semantics.
