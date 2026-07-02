The idea: ThinTensor

A model format where weights, KV cache, and execution graph are all stored/planned together.

Not:

tensor_name -> raw tensor bytes

But:

token step -> exact pages needed -> exact packed layout -> exact cache policy

The file is basically a compressed executable memory image for inference.

The real speed rule

For local LLM inference, especially batch-1 decode:

speed ≈ how few bytes you move per token

So ThinTensor should optimize:

weight bytes read
+ KV-cache bytes read/write
+ activation temporary memory
+ dequant overhead
+ page faults / copies

Not just “load model faster.”

Killer novelty 1: weight pages in decode order

Current formats mostly store tensors as named blobs.

ThinTensor stores them as execution pages:

page_000: layer_0_rmsnorm + qkv packed tile
page_001: layer_0_attn_out packed tile
page_002: layer_0_gate_up packed tile
page_003: layer_0_down packed tile
page_004: layer_1_rmsnorm + qkv packed tile
...

Each page is already packed for the target kernel:

cuda_sm89_q4_tile
cuda_sm120_fp4_tile
avx2_q4_tile
vulkan_q5_tile
metal_q4_tile

So the runtime does not convert/repack on load.

Killer novelty 2: mixed “truth budget” quantization

Do not quantize the whole model equally.

Each layer gets a memory budget based on sensitivity:

embedding: Q6
early attention: Q5
middle MLP: Q3/Q4
late attention: Q5
lm_head: Q6 or shared with embedding
outlier channels: sidecar FP16

So instead of a dumb Q4_K_M whole model, you store:

base matrix: 3/4 bit
outlier rows: 8/16 bit
important columns: 6 bit
scales: shared per superblock

This can be more memory-efficient than normal quant because you stop wasting bits on layers that do not need them.

Killer novelty 3: built-in KV cache codec

This is the huge one.

For long context, KV cache becomes a massive memory sink. TensorRT-LLM already exposes FP8/NVFP4 KV cache options, and KIVI showed 2-bit KV quantization can cut peak memory while increasing serving throughput. Google’s TurboQuant claims at least 6x KV memory reduction and up to 8x faster attention-logit computation with low-bit KV compression.

So ThinTensor should include a KV cache format, not just weight format:

recent tokens: FP8 / FP16
middle context: 3-4 bit compressed
old context: semantic summary pages
attention sinks: preserved high precision
rarely attended spans: CPU / SSD cold pages

KV layout:

K cache:
  per-channel quant
  rotated / transformed blocks
  3-4 bit packed

V cache:
  per-token quant
  4-8 bit depending on layer/head

special:
  first 4 tokens high precision
  last 256 tokens high precision
  retrieved spans high precision

This is where you actually beat existing stuff on memory.

Killer novelty 4: token-adaptive memory

Most formats are static. ThinTensor should be dynamic.

At runtime it tracks:

which layers are bandwidth-heavy
which attention heads are active
which KV blocks are being attended to
which pages are cold

Then it can do:

keep recent KV on GPU
compress older KV
evict dead blocks
prefetch likely next pages
skip cold MoE experts

For MoE models, this is insane:

expert_17 hot -> keep GPU
expert_04 warm -> CPU pinned
expert_29 cold -> compressed/offloaded

You do not store a MoE like “all experts equal.” You store it like a cache hierarchy.

Killer novelty 5: fused tensor pages

Instead of separate tensors:

q_proj.weight
k_proj.weight
v_proj.weight
gate_proj.weight
up_proj.weight

ThinTensor stores:

qkv_fused.weight
gate_up_fused.weight

And stores them already interleaved in the order the kernel reads them.

Example:

[layer 12 page]
rmsnorm weight
qkv packed tile
qkv scales
rope constants
attn out tile
gate_up packed tile
gate_up scales
down packed tile

This reduces metadata lookups, tiny allocations, random reads, and repacking.

Killer novelty 6: activation memory plan

The format stores a static scratch-buffer plan:

scratch_A: hidden state
scratch_B: qkv temp
scratch_C: mlp temp
reuse scratch_B after attention
reuse scratch_C after down_proj

So runtime does not allocate tensors layer by layer.

Memory becomes:

weights: compressed pages
KV: compressed/paged
activations: fixed ring buffer

This is way cleaner for small GPUs.

File layout
THINTENSOR v0

[model]
arch = llama/qwen/mistral
hidden_size
layers
heads
kv_heads
rope
tokenizer_hash
chat_template

[execution_tape]
embed
for layer in layers:
  rmsnorm
  qkv_fused
  rope
  paged_attention
  attn_out
  rmsnorm
  gate_up_fused
  silu_mul
  down
lm_head

[weight_pages]
page_id
layer
op
offset
compressed_size
original_shape
quant_scheme
kernel_layout
importance_score
checksum

[kv_codec]
recent_window = 256
sink_tokens = 4
k_quant = per_channel_q3/q4
v_quant = per_token_q4/q8
cold_policy = cpu_pinned/compressed
eviction_policy = attention_score + recency

[memory_plan]
gpu_resident_pages
cpu_pinned_pages
scratch_buffers
prefetch_distance
max_vram_targets = 4GB/6GB/8GB/12GB

[runtime_variants]
cuda_sm89
cuda_sm120
vulkan
metal
cpu_avx2
cpu_avx512
How it beats current formats
Thing	Current style	ThinTensor style
Weights	Named tensors	Execution-ordered pages
Quantization	Mostly one scheme	Per-layer/per-channel truth budget
KV cache	Runtime feature	Built into model format
Loading	Load tensors	Load only needed pages
Kernels	Runtime repacks	Stored in kernel-native layout
Activations	Framework allocs	Static scratch memory plan
MoE	All experts similar	Expert hot/cold memory tiers
The craziest version

The model file has multiple memory personalities:

profile_4gb:
  weights Q3/Q4
  KV q3
  CPU offload aggressive

profile_8gb:
  weights Q4/Q5
  KV q4/fp8
  last 512 tokens high precision

profile_16gb:
  weights Q5/Q6
  KV fp8/q8
  no CPU offload

Same file. Runtime picks based on your GPU.

Practical MVP

Build it as a sidecar first:

model.safetensors
model.thin.json
model.thin.idx
model.thin.kv

MVP features:

1. Read HF safetensors
2. Fuse q/k/v and gate/up
3. Reorder tensors by execution order
4. Pack into 2MB aligned pages
5. Add mixed quant per layer
6. Add static scratch-buffer plan
7. Run through a tiny custom loader
