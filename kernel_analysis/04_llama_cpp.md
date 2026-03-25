# llama.cpp: Memory-Bound Kernel Analysis

---

Here is the exhaustive analysis of llama.cpp's memory-bound CUDA kernels.

## 1. RMSNorm

**File**: `/home/kunwar/Work/kernel_libs/llama.cpp/ggml/src/ggml-cuda/norm.cu` lines 74-151

**Thread Block Configuration**:
- `ncols < 1024`: block_size = **256**, grid = `(nrows, nchannels, nsamples)`
- `ncols >= 1024`: block_size = **1024**, grid = `(nrows, nchannels, nsamples)`
- Template parameter `block_size` used for loop stride

**Thread-to-Data Mapping**: 1 thread block per row. Each thread handles columns `col = tid, tid+block_size, tid+2*block_size, ...` (strided across the row).

**Reduction Strategy**: Two-phase block_reduce:
1. `warp_reduce_sum` via `__shfl_xor_sync(0xffffffff, x, offset)` for offsets 16,8,4,2,1
2. If block_size > WARP_SIZE: warp lane 0 writes to `shared_vals[warp_id]`, syncthreads, then lanes < (block_size/WARP_SIZE) load from shared and do a second warp_reduce

**Shared Memory**: `32 * sizeof(float) = 128 bytes` when block_size > WARP_SIZE (for inter-warp reduction). Zero when block_size == WARP_SIZE.

**Two passes over the row**:
1. Pass 1: Accumulate `sum(x[col]^2)` into thread-local `tmp`
2. block_reduce SUM -> compute `scale = rsqrtf(mean + eps)`
3. Pass 2: Write `dst[col] = scale * x[col]` (optionally `* mul[col] + add[col]` for fused variant)

**Vector Loads**: None -- scalar f32 loads. No float4 vectorization.

**Fused Variants**: `rms_norm_f32<block_size, do_multiply, do_add>` fuses weight multiplication and bias addition into the same kernel. Uses `fastmodulo()` for broadcasting.

**Elements Per Thread**: `ncols / block_size` elements per thread.

## 2. LayerNorm

**File**: `/home/kunwar/Work/kernel_libs/llama.cpp/ggml/src/ggml-cuda/norm.cu` lines 4-38

**Thread Block Configuration**:
- `ncols < 1024`: block_size = **WARP_SIZE (32)**
- `ncols >= 1024`: block_size = **1024**
- Grid = `(nrows, nchannels, nsamples)`

**Thread-to-Data Mapping**: Identical to RMSNorm -- 1 block per row, strided threads.

**Reduction Strategy**: `block_reduce<SUM>` on a `float2` (mean_var.x for sum, mean_var.y for sum of squares). Same warp shuffle + shared memory two-phase pattern.

**Key difference from RMS**: Computes both mean AND variance in a single pass using `float2` accumulator, then `mean = sum_x / ncols`, `var = sum_x2/ncols - mean*mean`, `inv_std = rsqrtf(var + eps)`.

**Shared Memory**: `32 * sizeof(float2) = 256 bytes` when block_size > WARP_SIZE.

## 3. GroupNorm

**File**: `/home/kunwar/Work/kernel_libs/llama.cpp/ggml/src/ggml-cuda/norm.cu` lines 41-72

**Thread Block Configuration**:
- `group_size < 1024`: block_size = **WARP_SIZE (32)**
- `group_size >= 1024`: block_size = **1024**
- Grid = `(num_groups, 1, 1)` -- 1 block per group

**Three passes**: (1) sum elements -> mean, (2) compute `(x-mean)^2` and write intermediate, (3) multiply by `rsqrtf(var+eps)`.

## 4. L2Norm

**File**: `/home/kunwar/Work/kernel_libs/llama.cpp/ggml/src/ggml-cuda/norm.cu` lines 239-271

Structurally identical to RMSNorm. Only difference: `scale = rsqrtf(fmaxf(tmp, eps * eps))` instead of `rsqrtf(tmp/ncols + eps)`.

## 5. Softmax

**File**: `/home/kunwar/Work/kernel_libs/llama.cpp/ggml/src/ggml-cuda/softmax.cu`

### Variant A: Row-per-block (small ncols, fits in shared memory)

**Template**: `soft_max_f32<use_shared, ncols_template, block_size_template, T>` (line 55)

**Thread Block Configuration**:
- `nth` starts at WARP_SIZE (32) and doubles until `>= ncols` or reaches **CUDA_SOFT_MAX_BLOCK_SIZE = 1024**
- Grid = `(ne01, ne02, ne03)` -- one block per row
- Specialized templates for ncols = **32, 64, 128, 256, 512, 1024, 2048, 4096**

**Shared Memory**: `(GGML_PAD(ncols, WARP_SIZE) + WARP_SIZE) * sizeof(float)`. The `WARP_SIZE` extra is for inter-warp reduction buffer. E.g., for ncols=4096: `(4096 + 32) * 4 = 16,512 bytes`.

**Three passes over the row** (all using `vals` cache in shared memory or dst):
1. Pass 1: `val = x[col]*scale + slope*mask[col]`; store to `vals[col]`; track `max_val`
2. `block_reduce<MAX>` -> global max
3. Pass 2: `exp(vals[col] - max_val)` -> accumulate sum, store back to `vals[col]`
4. `block_reduce<SUM>` -> global sum
5. Pass 3: `dst[col] = vals[col] * inv_sum`

**When shared memory exceeds device limit** (`nbytes_shared > smpbo`): Falls back to `use_shared=false`, which uses `dst` as scratch space instead of shared memory, with only `WARP_SIZE * sizeof(float) = 128 bytes` of shared memory for the reduction buffer.

### Variant B: Cooperative kernel for very wide rows (line 141-317)

**Trigger**: `ncols > smpbo/sizeof(float)` AND `ncols/(ne01*ne02*ne03) > 8192` AND no mask/sinks/scaling.

**Launch**: Cooperative launch with `gridDim.x = nsm` (number of SMs), block_size = `8 * WARP_SIZE = 256`. Uses `__launch_bounds__(256, 1)`.

**Strategy**: Grid-sync across CTAs. Each CTA processes a strided subset of columns. `n_elem_per_thread = 4` registers per thread.
- Phase 1: CTA-local max -> GMEM tmp_maxs -> grid.sync() -> global max from CTA maxs
- Phase 2: exp + CTA-local sum -> GMEM tmp_sums -> grid.sync() -> global sum
- Phase 3: divide and write

**Shared Memory**: `32 * sizeof(float) = 128 bytes` (for block_reduce within each CTA).

### Softmax Back (line 250-270)

Block_size = **WARP_SIZE (32)**. Grid = `(nrows, 1, 1)`. Pure warp-level reduction, no shared memory.

## 6. Scale (Elementwise Multiply + Bias)

**File**: `/home/kunwar/Work/kernel_libs/llama.cpp/ggml/src/ggml-cuda/scale.cu`

**Thread Block Configuration**: Block_size = **CUDA_SCALE_BLOCK_SIZE = 256**. Grid = `min(0x7FFFFFFF, ceil(nelements/256))`.

**Pattern**: Flat 1D grid-stride loop. `dst[i] = scale * x[i] + bias`. No reduction, no shared memory.

## 7. Unary Ops (gelu, silu, relu, sigmoid, etc.)

**File**: `/home/kunwar/Work/kernel_libs/llama.cpp/ggml/src/ggml-cuda/unary.cu`

**Thread Block Configuration**: Block_size = **CUDA_NEG_BLOCK_SIZE = 256** for all unary ops. Grid = `ceil(k/256)`.

**Pattern**: Flat 1D, 1 element per thread. `dst[i] = op(x[i])`. No vectorization, no shared memory.

**Gated variants** (SwiGLU, GeGLU, ReGLU): Same block_size **CUDA_GLU_BLOCK_SIZE = 256**. Computes `op(x[j0]) * g[j1]` with stride-aware indexing.

## 8. Binary Broadcast Ops (add, mul, sub, div)

**File**: `/home/kunwar/Work/kernel_libs/llama.cpp/ggml/src/ggml-cuda/binbcast.cu`

**Thread Block Configuration**: Block_size = **128** (line 260). 3D block/grid decomposition:
```
block_dims.x = min(ne0/2, 128)
block_dims.y = min(ne1, 128/block_dims.x)
block_dims.z = min(ne2*ne3, min(128/(x*y), 64))
```

**Fallback**: When `block_nums.z > 65535` or `block_nums.y > 65535`, uses `k_bin_bcast_unravel` with flat 1D decomposition and `fastdiv`/`fastmodulo` for index recovery.

**Pattern**: 1 element per thread + optional grid-stride loop over ne0 dimension. Scalar loads. Uses `fastmodulo` for broadcasting dimensions.

## 9. Row Reductions (sum_rows)

**File**: `/home/kunwar/Work/kernel_libs/llama.cpp/ggml/src/ggml-cuda/reduce_rows.cuh` and `sumrows.cu`

**Thread Block Configuration**:
- If `nrows/nsm < 2` (few rows, need to keep SMs busy): block_size = **512**
- Else if `ncols < 1024`: block_size = **32** (one warp)
- Else: block_size = **128**
- Grid = `(nrows, 1, 1)`

**Optimization**: **8x manual unroll** in the accumulation loop. Uses 8 separate `sum_temp[8]` accumulators to hide latency, then combines them.

**Reduction**: `block_reduce<SUM>` using shared memory (`float shared_vals[32]`).

## 10. RoPE (Rotary Position Embeddings)

**File**: `/home/kunwar/Work/kernel_libs/llama.cpp/ggml/src/ggml-cuda/rope.cu`

**Thread Block Configuration**: **CUDA_ROPE_BLOCK_SIZE = 256**. Block dims = `(1, 256, 1)`. Grid = `(nr, n_blocks_x, 1)` where `nr = total_rows` and `n_blocks_x = ceil(ne00 / (2*256))`.

**Thread-to-Data Mapping**: 2D mapping. `threadIdx.y` maps to dimension pairs (each thread handles 2 consecutive elements), `blockIdx.x` maps to rows.
- `i0 = 2 * (blockDim.y * blockIdx.y + threadIdx.y)` -- the dimension index (processes pairs)
- `row_dst = blockDim.x * blockIdx.x + threadIdx.x` -- the row (blockDim.x=1, so each block handles 1 row)

**Vector Stores**: For `rope_norm`, uses coalesced 8-byte writes via `float2` or `half2`:
```cpp
float2 v = make_float2(x0, x1);
ggml_cuda_memcpy_1<8>(dst + idst, &v);  // 8 bytes = float2
```

**No reduction, no shared memory**. Pure elementwise with trig computation.

**Variants**: `rope_norm` (standard), `rope_neox` (GPT-NeoX interleaving: x0 and x1 separated by n_dims/2), `rope_multi` (multi-modal with sections), `rope_vision`.

**Fused variant**: `rope_fused` combines ROPE + VIEW + SET_ROWS using `row_indices` indirection.

## 11. MatVec Float (GEMV for f32/f16/bf16)

**File**: `/home/kunwar/Work/kernel_libs/llama.cpp/ggml/src/ggml-cuda/mmvf.cu`

**Thread Block Configuration**:
- Adaptive block_size selection (lines 426-438): starts at `warp_size`, tries all multiples of `warp_size` up to **256** (or **128** on AMD GCN/CDNA), picks whichever minimizes `niter = ceil(ncols / (2*block_size))`
- Grid = `(nrows, nchannels_dst, nsamples_or_ntokens)` -- **1 block per output row**
- ncols_dst template: 1 through 8 (batched GEMV)

**Vector Loads**: **float2 (8 bytes)** for f32, **half2 (4 bytes)** for f16, **nv_bfloat162** for bf16. The inner loop reads `ncols/2` pairs:
```cpp
const float2 * x2 = (const float2 *) x;
for (int col2 = tid; col2 < ncols2; col2 += block_size) {
    const float2 tmpx = x2[col2];
    for (int j = 0; j < ncols_dst; ++j) {
        const float2 tmpy = y2[j*stride_col_y2 + col2];
        ggml_cuda_mad(sumf[j], tmpx.x, tmpy.x);
        ggml_cuda_mad(sumf[j], tmpx.y, tmpy.y);
    }
}
```

**Reduction Strategy**: Two-level:
1. `warp_reduce_sum<warp_size>(sumf[j])` -- warp shuffle
2. If `block_size > warp_size`: write to `buf_iw[tid/warp_size]`, syncthreads, load back, second warp_reduce_sum

**Shared Memory**: `warp_size * sizeof(float)` for inter-warp reduction = **128 bytes** (or 2x if fused with gate).

**Batching**: Template parameter `ncols_dst` (1-8). Each thread maintains `sumf[ncols_dst]` accumulators. All dst columns share the same x row, only y pointers differ.

**Fusion**: Can fuse gate (SwiGLU, GeGLU) and bias addition into the GEMV kernel, avoiding extra memory passes.

## 12. MatVec Quantized (GEMV for quantized weights)

**File**: `/home/kunwar/Work/kernel_libs/llama.cpp/ggml/src/ggml-cuda/mmvq.cu`

**Thread Block Configuration**: Block dims = `(warp_size, nwarps, 1)`. Grid = `(ceil(nrows/rows_per_block), nchannels_dst, nsamples_or_ntokens)`.

The **nwarps** value is GPU-architecture and type-dependent:

| ncols_dst | NVIDIA (generic) | AMD GCN | AMD RDNA3/4 (simple types) |
|-----------|-------------------|---------|---------------------------|
| 1         | 4                 | 2       | 8                         |
| 2-4       | 4                 | 2       | 1                         |
| 5-8       | 2                 | 1       | 1                         |

Total threads per block = `nwarps * warp_size`. E.g., NVIDIA ncols_dst=1: `4 * 32 = 128 threads`.

**rows_per_block**: NVIDIA/GCN: 1 for ncols_dst=1 (or nwarps for small_k), 2 for ncols_dst=2-8. RDNA: always 1.

**Thread-to-Data Mapping** (key formula, line 300):
```cpp
constexpr int blocks_per_iter = vdr * nwarps * warp_size / qi;
for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
    const int kqs = vdr * (tid % (qi/vdr));
    // each thread processes kqs offset within quant block kbx
}
```

**VDR (Vector Dot Ratio)** per type -- how many int32 values are loaded per vec_dot call:

| Type | VDR | qk | qi | Elements per vec_dot |
|------|-----|----|----|---------------------|
| Q4_0 | 2   | 32 | 4  | 64 elements          |
| Q4_1 | 2   | 32 | 4  | 64 elements          |
| Q5_0 | 2   | 32 | 4  | 64 elements          |
| Q8_0 | 2   | 32 | 8  | 64 elements          |
| Q2_K | 1   | 256| 16 | 256 elements         |
| Q3_K | 1   | 256| 16 | 256 elements         |
| Q4_K | 2   | 256| 8  | 512 elements         |
| Q5_K | 2   | 256| 8  | 512 elements         |
| Q6_K | 1   | 256| 8  | 256 elements         |

Where `qi = qk / (4 * qr)` represents the number of int32s per quant block.

**Reduction Strategy**: Three-level:
1. Each thread accumulates `tmp[ncols_dst][rows_per_block]` partial sums
2. Non-warp-0 writes to `__shared__ float tmp_shared[nwarps-1][ncols_dst][rows_per_block][warp_size]`
3. Warp 0 sums across warps, then `warp_reduce_sum`
4. `threadIdx.x < rows_per_block` writes output

**Shared Memory**: `(nwarps-1) * ncols_dst * rows_per_block * warp_size * sizeof(float)`. E.g., NVIDIA ncols_dst=1: `3 * 1 * 1 * 32 * 4 = 384 bytes`. With gate fusion: 2x.

**Batching**: Template `ncols_dst` from 1 to **MMVQ_MAX_BATCH_SIZE = 8**. Inner loop processes all dst columns per quant block access (reuses the dequantized x data).

**Small-K optimization** (line 493-510): When K is small enough that a single iteration covers all quant blocks (`blocks_per_row_x < nwarps * blocks_per_iter_1warp`), increases `rows_per_block = nwarps` so each warp handles a different row, improving occupancy.

**Fusion**: Like mmvf, supports fused gate (SwiGLU/GeGLU) and bias in the same kernel.

## 13. Argmax (Row-wise Reduction)

**File**: `/home/kunwar/Work/kernel_libs/llama.cpp/ggml/src/ggml-cuda/argmax.cu`

**Thread Block Configuration**: `num_threads = min(1024, ceil(ne00/32)*32)`. Grid = `(nrows, 1, 1)`.

**Reduction**: Manual `__shfl_xor_sync` carrying both maxval AND argmax. Two-level: warp shuffle -> shared memory `shared_maxval[max_warps]` + `shared_argmax[max_warps]` -> second warp shuffle. Uses `max_warps = 1024/32 = 32`.

## 14. Cross-Entropy Loss (Row Reduction with Softmax)

**File**: `/home/kunwar/Work/kernel_libs/llama.cpp/ggml/src/ggml-cuda/cross-entropy-loss.cu`

**Thread Block Configuration**: block_size = **WARP_SIZE (32)**. Grid = `(nrows, 1, 1)`.

**Three passes**: (1) find max via `warp_reduce_max`, (2) exp + sum via `warp_reduce_sum`, (3) log-softmax dot product with labels.

**Shared Memory**: When `use_shared=true`, uses `extern __shared__ float tmp[]` to cache logits between passes (size = nclasses * sizeof(float)).

---

## Summary Table of Block Sizes

| Kernel | Block Size | Grid Mapping | Shared Memory | Vectorization | Reduction |
|--------|-----------|--------------|---------------|---------------|-----------|
| RMSNorm | 256/1024 | 1 block/row | 128B | None | block_reduce(warp_shuffle + shmem) |
| LayerNorm | 32/1024 | 1 block/row | 256B (float2) | None | block_reduce on float2 |
| Softmax | 32-1024 (power of 2) | 1 block/row | ncols*4 + 128B | None | block_reduce MAX then SUM |
| Softmax (coop) | 256 | nsm blocks, loops rows | 128B | 4 elem/thread registers | grid_sync + block_reduce |
| Scale | 256 | flat 1D | None | None | None |
| Unary ops | 256 | flat 1D | None | None | None |
| Binary bcast | 128 | 3D (ne0/2, ne1, ne2*ne3) | None | None | None |
| Sum rows | 32/128/512 | 1 block/row | 128B | None | block_reduce SUM (8x unroll) |
| RoPE | 256 (threadIdx.y) | (rows, ceil(ne00/512)) | None | float2/half2 stores | None |
| GEMV float | 32-256 (adaptive) | 1 block/row | 128B | float2/half2/bf162 loads | warp_shuffle + shmem |
| GEMV quant | 128-256 (nwarps*32) | rows/rpb, channels, samples | 384B+ | Quant-block vec_dot | warp_shuffle + shmem (multi-warp) |
| Argmax | up to 1024 | 1 block/row | 256B | None | warp_shuffle carrying val+idx |