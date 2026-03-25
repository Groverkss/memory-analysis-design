# NVIDIA Apex: Memory-Bound Kernel Analysis

## Source: https://github.com/NVIDIA/apex.git

---

## 1. LAYER NORM (Basic) -- `csrc/layer_norm_cuda_kernel.cu`

**File**: `csrc/layer_norm_cuda_kernel.cu`

### Thread Block Configuration
- **Block size**: `dim3(32, 4, 1)` = 128 threads
- **Grid**: `dim3(1, min(n1, maxGridY), 1)` -- one block column per row of input
- `blockDim.x = 32` (one warp), `blockDim.y = 4` (4 warps per block)
- **Each block normalizes one or more rows** (rows are n1 dimension, columns are n2 = hidden dim)

### Thread-to-Data Mapping
- Each thread's linear ID: `thrx = threadIdx.x + threadIdx.y * blockDim.x`
- For the Welford pass, thread `thrx` starts at element `4*thrx` and strides by `4*numx` where `numx = blockDim.x * blockDim.y = 128`
- For fp16 specialization: starts at `8*thrx`, strides by `8*numx`, uses `__half2` vectorized loads
- For the output pass: thread `thrx` writes elements `thrx, thrx+numx, thrx+2*numx, ...` up to n2

### Vector Loads
- **fp32**: loads 4 consecutive elements per iteration via scalar loop `for k=0..3`
- **fp16**: loads 8 elements per iteration using `__half22float2` -- effectively `float2` = 4 bytes = 2 fp16 elements per intrinsic, 4 such loads = 8 elements
- Alignment check: if `lvals` is not 32-bit aligned, thread 0 handles the first element separately

### Reduction Strategy
- **Welford online algorithm** for numerically stable mean/variance
- **Intra-warp**: `WARP_SHFL` with lane offsets `1, 2, 4, 8, 16` (5 iterations), using `cuChanOnlineSum` to merge partial Welford stats
- **Inter-warp** (when `blockDim.y > 1`): shared memory reduction, halving offset from `blockDim.y/2` down to 1
- Final broadcast: `WARP_SHFL(mu, 0)` or via shared memory `ubuf[0]`

### Shared Memory Usage
- `nshared = threads.y * sizeof(U) + (threads.y/2) * sizeof(U)` = `4*4 + 2*4 = 24 bytes` for float
- Used for inter-warp Welford merge (storing mu, sigma2, count between warps)

### Elements Per Thread
- Welford: `n2/128` elements per thread (both fp32 and fp16)
- Output: `n2/128` elements per thread

### Backward Gradient Kernels
- **grad_gamma/grad_beta** (`cuComputePartGradGammaBeta`): block `dim3(32, 4)`, grid `dim3(ceil(n2/32), 16)`, shared mem = `2 * sizeof(U) * 4 * 4 * (32+1)` = 4224 bytes. Uses `part_size=16` partial reductions.
- **grad_input** (`cuComputeGradInput`): same `dim3(32, 4)` block, one block per row. Also loads 4 elements at a time with stride `4*numx`.

---

## 2. LAYER NORM (Contrib/Optimized) -- `apex/contrib/csrc/layer_norm/`

**Files**:
- `apex/contrib/csrc/layer_norm/ln_kernel_traits.h`
- `apex/contrib/csrc/layer_norm/ln_fwd_kernels.cuh`
- `apex/contrib/csrc/layer_norm/ln_fwd_cuda_kernel.cu`

### Architecture
This is the **production-quality** layer norm with extensive specialization. Every hidden size gets hand-tuned parameters.

### Key Configuration Parameters
Format: `REGISTER_FWD_LAUNCHER(HIDDEN_SIZE, WTYPE, ITYPE, OTYPE, CTYPE, CTAS_PER_ROW, WARPS_M, WARPS_N, BYTES_PER_LDG)`

| Hidden Size | CTAS_PER_ROW | WARPS_M | WARPS_N | BYTES_PER_LDG | Threads/CTA |
|------------|-------------|---------|---------|--------------|-------------|
| 768-2304   | 1           | 4       | 1       | 16           | 128         |
| 3072-8192  | 1           | 1       | 4       | 16           | 128         |
| 3840       | 1           | 1       | 4       | **4**        | 128         |
| 10240      | 1-2 (fp32:2)| 1       | 4       | 16           | 128         |
| 12288      | 2           | 1       | 4       | 16           | 128         |
| 12800      | 2           | 1       | 4       | **4**         | 128         |
| 16384      | 2           | 1       | 4       | 16           | 128         |
| 18432      | 2-4         | 1       | 4       | 16           | 128         |
| 24576      | 2-4         | 1       | 4       | 16           | 128         |
| 32768      | 4           | 1       | 4       | 16           | 128         |
| 49152      | 4-8         | 1       | 4       | 16           | 128         |
| 65536      | 8           | 1       | 4       | 16           | 128         |

**Key design decisions**:
- **Small hidden (<=2304)**: `WARPS_M=4, WARPS_N=1` -- 4 rows per CTA, single warp per row. Each row fits in one warp's work.
- **Medium hidden (3072-8192)**: `WARPS_M=1, WARPS_N=4` -- 1 row per CTA, 4 warps cooperate on one row (need inter-warp reduction).
- **Large hidden (>=10240)**: Multiple CTAs per row (`CTAS_PER_ROW=2,4,8`) with cooperative kernel launch and inter-CTA synchronization via global memory barriers.
- **Always 128 threads/CTA** (`WARPS_M * WARPS_N * 32`).

### Thread-to-Data Mapping (from `ln_kernel_traits.h`)
- `NUM_ELTS = BYTES_PER_LDG / sizeof(input_t)` -- for fp16 + 16B load = 8 elements; for fp32 + 16B = 4 elements
- `VEC_COLS = COLS / ELTS_PER_LDG` = number of vectorized columns
- `VEC_COLS_PER_LDG = CTAS_PER_ROW * THREADS_PER_ROW` = threads that load simultaneously
- `LDGS = VEC_COLS / VEC_COLS_PER_LDG` = number of vectorized loads per thread

Example: hidden=4096, fp16, BYTES_PER_LDG=16 -> NUM_ELTS=8, THREADS_PER_ROW=128, VEC_COLS=4096/8=512, LDGS=512/128=4. Each thread does 4 x 16-byte loads = 4*8 = 32 fp16 elements.

### Vector Loads
- Uses `Vec<Elt_type, NUM_ELTS>` abstraction backed by `uint4` (16 bytes), `uint8` (32 bytes), or `uint16` (64 bytes)
- Default: **16 bytes per load** (`BYTES_PER_LDG=16`)
- Non-power-of-2 hidden sizes (3840, 12800, 25600, 30720) use **4 bytes per load** to handle non-aligned boundaries
- Load via: `this->data.vec = static_cast<const Vec_type*>(base_ptr)[idx]` -- direct vectorized cast

### Reduction Strategy (from `ln_utils.cuh`)
Three levels:
1. **Intra-warp**: `warp_shuffle_xor` with all 5 log2(32) steps
2. **Inter-warp** (WARPS_N > 1): shared memory write by warp leaders, sequential read by all
3. **Inter-CTA** (CTAS_PER_ROW > 1): global memory workspace + `spin_wait_` barrier using `red.release.gpu.global.add.s32` and `ld.global.acquire.gpu.b32`

### Occupancy
- `cudaOccupancyMaxActiveBlocksPerMultiprocessor` is called to determine `ctas_per_col`
- `ctas_per_col = multiProcessorCount * ctas_per_sm / CTAS_PER_ROW`
- Dynamic shared memory >= 48KB triggers `cudaFuncSetAttribute` for max dynamic shared mem

---

## 3. SOFTMAX (Megatron) -- `csrc/megatron/`

**Files**:
- `csrc/megatron/scaled_masked_softmax.h`
- `csrc/megatron/scaled_upper_triang_masked_softmax.h`

### The Fundamental Design: Warp-Level Softmax

All softmax elements for one row fit in registers across one warp. The dimension is rounded to `next_power_of_two = 1 << log2_ceil(element_count)`, up to **16384**.

### Compile-Time Constants (parameterized by `log2_elements`)
```
next_power_of_two = 1 << log2_elements
WARP_SIZE = min(next_power_of_two, 32)
WARP_ITERATIONS = next_power_of_two / WARP_SIZE
WARP_BATCH = (next_power_of_two <= 128) ? 2 : 1
ELEMENTS_PER_LDG_STG = (WARP_ITERATIONS < 4) ? 1 : 4
```

### Concrete Examples

| seq_len | next_pow2 | WARP_SIZE | WARP_ITER | WARP_BATCH | ELEM/LDG | elts/thread |
|---------|-----------|-----------|-----------|------------|----------|-------------|
| 1-32    | 1-32      | 1-32      | 1         | 2          | 1        | 1*2=2       |
| 33-64   | 64        | 32        | 2         | 2          | 1        | 2*2=4       |
| 65-128  | 128       | 32        | 4         | 2          | 4        | 4*2=8       |
| 129-256 | 256       | 32        | 8         | 1          | 4        | 8           |
| 257-512 | 512       | 32        | 16        | 1          | 4        | 16          |
| 513-1024| 1024      | 32        | 32        | 1          | 4        | 32          |
| 2049-4096| 4096     | 32        | 128       | 1          | 4        | 128         |
| 8193-16384| 16384   | 32        | 512       | 1          | 4        | 512         |

### Thread Block Configuration
- **Always 128 threads per block** (`constexpr int threads_per_block = 128`)
- `warps_per_block = threads_per_block / warp_size`
- `batches_per_block = warps_per_block * batches_per_warp`
- Grid: `dim3(query_seq_len / batches_per_block, attn_heads, batches)` for scaled_softmax
- For small element counts (<=32), warp_size < 32, so multiple "sub-warps" per physical warp

### Thread-to-Data Mapping
- `local_idx = threadIdx.x` (within logical warp)
- `thread_offset = first_batch * element_count + ELEMENTS_PER_LDG_STG * local_idx`
- Each thread reads elements at positions: `ELEMENTS_PER_LDG_STG * local_idx + it * WARP_SIZE` for `it = 0..WARP_ITERATIONS`

### Vector Loads
- `copy_vector<input_t, ELEMENTS_PER_LDG_STG>` where ELEMENTS_PER_LDG_STG is 1 or 4
- For fp16 with 4 elements: `*((float2*)dst) = *((float2*)src)` = single 8-byte load for 4 fp16 values
- For bf16 with 4 elements: same `float2` trick

### Reduction Strategy
- **Pure warp shuffle** using `__shfl_xor_sync` (no shared memory for the reduction itself)
- `warp_reduce<acc_t, WARP_BATCH, WARP_SIZE, Max>` for max
- `warp_reduce<acc_t, WARP_BATCH, WARP_SIZE, Add>` for sum
- `log2(WARP_SIZE)` shuffle steps

### Shared Memory: **Zero** (no shared memory used)

### Edge Cases
- Elements beyond `element_count` set to `-inf` (for max) or 0 (for output)
- Batches that don't align to `WARP_BATCH`: checked via `local_batches = min(micro_batch_size - first_batch, WARP_BATCH)`
- Masked positions: set to `-10000.0` (scaled_masked) or `-inf` (upper_triang)

---

## 4. BATCH NORM (Welford-based) -- `csrc/welford.cu`

**File**: `csrc/welford.cu`

### Constants
```cpp
#define ELEMENTS_PER_ITER 4      // concurrency per thread to hide latency
#define ELEMENTS_PER_THREAD 16
#define OPTIMAL_TILE_W 32
#define MAX_H_BLOCK 128
#define MAX_BLOCK_SIZE 512
```

### Launch Configuration (`flexible_launch_configs`)
```cpp
block_x = min(h_last_pow2(stride), OPTIMAL_TILE_W=32)
block_y = min(h_last_pow2(div_ru(reduction, ELEMENTS_PER_THREAD=16)), MAX_BLOCK_SIZE/block_x)
// if not at MAX_BLOCK_SIZE, expand block_x
if (block_x * block_y != MAX_BLOCK_SIZE)
    block_x = min(h_last_pow2(stride), MAX_BLOCK_SIZE / block_y)
grid_x = div_ru(stride, block_x)
grid_y = min(div_ru(reduction, block_y * ELEMENTS_PER_THREAD), MAX_H_BLOCK=128)
```

### Welford Kernel (NHWC layout)
- `PARALLEL_LOADS = 4` -- each thread maintains 4 independent Welford accumulators to hide memory latency
- Thread mapping: `c_offset = blockIdx.x * blockDim.x + threadIdx.x` (channel), `m_offset = blockIdx.y * blockDim.y + threadIdx.y` (spatial)
- Grid reduction: when `gridDim.y > 1`, uses staging global memory + `atomicAdd` semaphore for cross-block sync
- Shared memory: `MAX_BLOCK_SIZE * sizeof(accscalar_t)` for each of mean, m2n, count = `512 * 3 * 4 = 6144 bytes`

---

## 5. GROUP NORM (One-Pass) -- `apex/contrib/csrc/group_norm/`

### Template Parameters
- `ACTS_PER_BLOCK_`: 64, 128, 256, 512 -- selected by HW threshold
- `CHANNELS_PER_GROUP_`: 4 through 160 (extensive list)
- `THREADS_PER_BLOCK_`: always 512

### ACTS_PER_BLOCK Selection
Based on `params.hw`:
- `hw >= 512`: ACTS_PER_BLOCK=512
- `hw >= 256`: ACTS_PER_BLOCK=256
- `hw >= 128`: ACTS_PER_BLOCK=128
- `hw >= 0`:   ACTS_PER_BLOCK=64

### Derived Constants
```cpp
CHANNELS_PER_THREAD = 2  // loads fp16x2 = one __half2
THREADS_PER_ACT = CHANNELS_PER_GROUP / CHANNELS_PER_THREAD
ACTS_PER_LOOP = THREADS_PER_BLOCK / THREADS_PER_ACT
ACTS_PER_THREAD = ceil(ACTS_PER_BLOCK / ACTS_PER_LOOP)
```

### Reduction
- `cub::BlockReduce<float2, THREADS_PER_BLOCK>` for sum and sum-of-squares
- For multi-CTA: double-buffered global memory reduction via spin_wait barrier

---

## 6. L2 NORM (Multi-Tensor) -- `csrc/multi_tensor_l2norm_kernel.cu`

### Constants
```cpp
BLOCK_SIZE = 512
ILP = 4  // instruction-level parallelism
```

### Vector Loads
```cpp
typedef typename std::aligned_storage<ILP * sizeof(T), ILP * alignof(T)>::type LT;
((LT*)dst)[dst_offset] = ((LT*)src)[src_offset];  // 4-element vectorized load
```
For fp32: 16-byte loads. For fp16: 8-byte loads.

### Reduction
- Each thread accumulates `ILP=4` squared values
- Block reduction via `reduce_block_into_lanes` using shared memory (`float s_vals[512]`)
- Output: atomic add to global `output[blockIdx.x]`

---

## Summary of Key Patterns for IREE

**Universal constants across all Apex kernels:**

| Pattern | Value | Rationale |
|---------|-------|-----------|
| Warp size | 32 | Hardware constant |
| Max block size | 128-512 | 128 for norm/softmax, 512 for BN/GN/L2 |
| Vector load width | 4, 8, or 16 bytes | Always a power of 2 |
| Elements per thread | 4-512 | Depends on reduction dim |

**Critical design patterns:**
1. **Reduction dim maps to WARP_SIZE with multi-iteration** -- Softmax: `WARP_ITERATIONS = next_pow2(n) / 32`, all data in registers
2. **Batch dim maps to blockDim.y** -- LayerNorm: `blockDim.y=4` warps, each warp handles one row
3. **Channel dim maps to blockIdx.x** -- BatchNorm: one block per channel
4. **PARALLEL_LOADS=4** hides memory latency -- Both Welford (BN) and GroupNorm use this
5. **Vector loads scale with type**: fp16 uses `__half2`/`float2` (4 elements/load), fp32 uses `float4` (4 elements/load), uint4 (16 bytes) is the maximum single-instruction load
6. **Occupancy-aware block count**: contrib LayerNorm uses `cudaOccupancyMaxActiveBlocksPerMultiprocessor` to set `ctas_per_col`
7. **Multi-CTA cooperation**: for hidden_size > ~8192, multiple CTAs per row with global memory barriers
8. **Persistent kernels**: NHWC BN keeps data in registers across outer loops, uses `DESIRED_OCCUPANCY` in `__launch_bounds__`
