# PyTorch: Memory-Bound Kernel Analysis

---

# PyTorch Memory-Bound CUDA Kernel Analysis

## 1. VECTORIZED MEMORY ACCESS INFRASTRUCTURE

**File:** `/home/kunwar/Work/kernel_libs/pytorch/aten/src/ATen/native/cuda/MemoryAccess.cuh`

### `aligned_vector` (line 181-184)
The fundamental vectorization primitive. A struct with `alignas(sizeof(scalar_t) * vec_size)` that forces the compiler to emit vector load/store instructions:

```cpp
template<typename scalar_t, int vec_size>
struct alignas(sizeof(scalar_t) * vec_size) aligned_vector {
  scalar_t val[vec_size];
};
```

### Vectorization Width Selection (lines 509-533)
`can_vectorize_up_to<scalar_t>(pointer)` checks pointer alignment to determine max vector width. Returns 8, 4, 2, or 1. On CUDA (not ROCm): vec8 if 8-aligned, vec4 if 4-aligned, vec2 if 2-aligned, else 1.

### Low-level Vector Loads (lines 620-646)
`ld_vec<Alignment>` uses inline PTX on NVIDIA:
- 16 bytes: `ld.global.v4.u32` (4x uint32)
- 8 bytes: `ld.global.v2.u32` (2x uint32)
- 4 bytes: `ld.global.u32` (1x uint32)

### Vectorized Policy (lines 318-369)
The `policies::vectorized<vec_size, data_t, elems_per_thread>` struct handles the load/store pattern for contiguous elementwise ops:
- `loop_size = elems_per_thread / vec_size`
- Each thread loads `loop_size` vectors of width `vec_size`
- Thread index: `thread_idx + i * num_threads()` where `i` iterates over `loop_size`

---

## 2. ELEMENTWISE KERNEL INFRASTRUCTURE

**Files:**
- `/home/kunwar/Work/kernel_libs/pytorch/aten/src/ATen/native/cuda/thread_constants.h`
- `/home/kunwar/Work/kernel_libs/pytorch/aten/src/ATen/native/cuda/CUDALoops.cuh`
- `/home/kunwar/Work/kernel_libs/pytorch/aten/src/ATen/native/cuda/Loops.cuh`

### Thread Constants (thread_constants.h)
```
CUDA:  num_threads() = C10_WARP_SIZE * 4 = 128,  thread_work_size() = 8
ROCm:  num_threads() = 256,                       thread_work_size() = 4
block_work_size() = num_threads() * thread_work_size()
  => CUDA: 128 * 8 = 1024 elements per block
  => ROCm: 256 * 4 = 1024 elements per block
```

### Vectorized Elementwise `elems_per_thread` (CUDALoops.cuh, lines 93-101)
Depends on IO size (sum of input + output element sizes):
- `io_sizes == 1` (e.g., uint8): `elems_per_thread = 16`
- `io_sizes > 1` (e.g., float): `elems_per_thread = 8`

On ROCm, further split: `io_sizes < 4` gets 8, `io_sizes >= 4` gets 4.

### Non-vectorized Elementwise
Uses `elementwise_thread_work_size() = 4` always (line 106).

### Grid Configuration (Loops.cuh, line 275)
```
grid = ceil(N / block_work_size())
```
where `block_work_size = io_block_work_size<io_sizes>() = num_threads() * elems_per_thread<io_sizes>()`.

### Thread-to-Data Mapping
For vectorized case (MemoryAccess.cuh `policies::vectorized`, lines 334-345):
```
For each loop iteration i in [0, loop_size):
  index = threadIdx.x + i * num_threads()
  Load vec_size scalars from &data[index * vec_size]
```
For non-vectorized (unroll policy, lines 268-281):
```
For each i in [0, elems_per_thread):
  linear_idx = threadIdx.x + i * num_threads + block_work_size * blockIdx.x
```
This is a **strided** pattern: threads in a warp access consecutive elements (coalesced), then stride by num_threads.

---

## 3. LAYER NORM KERNEL

**File:** `/home/kunwar/Work/kernel_libs/pytorch/aten/src/ATen/native/cuda/layer_norm_kernel.cu`

### Architecture Overview
Two paths:
1. **Vectorized path** (fast): Used when all buffers are aligned, N is a multiple of `vec_size=4`, dtype is float/half/bf16, N <= 2^24.
2. **Fallback path** (scalar): Two-kernel approach -- first compute moments, then apply normalization.

### Vectorized Forward Kernel Launch (lines 1030-1049)
```
threads = (warp_size, num_threads() / warp_size, 1) = (32, 4, 1)  [128 threads]
blocks  = (M,)   // one block per row
shared_memory = threads.y > 1 ? threads.y * 3/2 * sizeof(T_ACC) : 0
              = 4 * 1.5 * 4 = 24 bytes (for float)
```
So: **128 threads per block, M blocks, one block per row.**

### Welford's Online Algorithm (lines 133-182)

**Per-element update** (`cuWelfordOnlineSum`, line 133):
```
delta = val - curr_sum.mean
new_count = curr_sum.count + 1
new_mean = curr_sum.mean + delta * (1/new_count)   // reciprocal, NOT division
return {new_mean, curr_sum.sigma2 + delta * (val - new_mean), new_count}
```
Key: uses `1.f/new_count` instead of proper division for speed, at slight accuracy cost.

**Parallel combine** (`cuWelfordCombine`, line 153):
```
delta = dataB.mean - dataA.mean
count = dataA.count + dataB.count
coef = 1/count
nA = dataA.count * coef; nB = dataB.count * coef
mean = nA*dataA.mean + nB*dataB.mean
sigma2 = dataA.sigma2 + dataB.sigma2 + delta*delta*dataA.count*nB
```

### Stats Computation (`compute_stats`, lines 184-244)
1. **Vectorized load**: Casts `X` to `vec_t*` (float4 for float), loads 4 elements at once.
2. **Per-thread Welford**: Each thread processes `ceil(N/vec_size / numx)` vectors, updating its local Welford state.
3. **Intra-warp reduction**: `WARP_SHFL_DOWN` for `log2(warp_size)` steps combining Welford states.
4. **Inter-warp reduction** (when `blockDim.y > 1`): Uses shared memory (`buf`). Upper half warps write, lower half merges, repeated for `log2(blockDim.y)` rounds.
5. Final: thread (0,0) writes `mean` and `sigma2/N` to shared memory, all threads read back.

### Forward Apply (lines 247-317)
Vectorized read of X, gamma, beta as `vec_t` (4 elements at a time). Each thread computes:
```
out[ii] = gamma[ii] * (rstd * (X[ii] - mean)) + beta[ii]
```
Writes back as `vec_t` for coalesced 128-bit stores.

### Fallback Path (lines 1116-1122)
```
RowwiseMomentsCUDAKernel<<<M, 512, 0, stream>>>(N, eps, X, mean, rstd)
LayerNormForwardCUDAKernel<<<M, 256, 0, stream>>>(N, X, mean, rstd, gamma, beta, Y)
```
512 threads for the reduction kernel (`kCUDABlockReduceNumThreads`), 256 for the elementwise kernel (`kCUDANumThreads`).

### Backward Gradient (lines 356-597)
Two versions: scalar (`compute_gI`) and vectorized (`layer_norm_grad_input_kernel_vectorized`).

The backward requires **two reductions** over N (for `stats_x1` = sum of `dY * gamma` and `stats_x2` = sum of `dY * gamma * (X - mean) * rstd`), followed by an elementwise pass. Uses `BlockReduceSum` from `block_reduce.cuh`.

The vectorized backward (line 462) loads X, dY, gamma as `vec_t` for the reduction pass, then reloads them for the elementwise output pass. The thread stride is `blockDim.x * vec_size`.

### GammaBeta Backward (lines 757-1015)
Block dimensions: `block_dim_x = 32`, `block_dim_y` varies with M:
- M < 64: `block_dim_y=1, rows_per_block_y=8`
- M < 128: `block_dim_y=8, rows_per_block_y=64`
- M < 256: `block_dim_y=16, rows_per_block_y=128`
- M >= 256: `block_dim_y=32, rows_per_block_y=256`

For M > 64K and small N: launches multi-block partial reduction followed by `.sum(0)`.

Uses **shared memory transpose** for the final reduction within each block (line 826-863): writes (dg_sum, db_sum) transposed into shared memory with +1 padding to avoid bank conflicts, then each warp reduces a full column via `WARP_SHFL_XOR`.

---

## 4. SOFTMAX KERNEL

**File:** `/home/kunwar/Work/kernel_libs/pytorch/aten/src/ATen/native/cuda/SoftMax.cu` and `PersistentSoftmax.cuh`

### Dispatch Strategy (host_softmax, lines 1066-1270)

**Case 1: inner_size == 1 (reduction along last dim)**

**Sub-case 1a: Small dim_size (<=2048, <=8KB)** -- Uses **persistent warp softmax** from `PersistentSoftmax.cuh`.
- Block size: 128 threads always (`threads_per_block = 128`)
- `WARP_SIZE = min(next_power_of_two(dim_size), 32)`
- `WARP_ITERATIONS = next_power_of_two / WARP_SIZE`
- `WARP_BATCH = (next_power_of_two <= 128) ? 2 : 1`
- Warps per block = 128 / WARP_SIZE
- Batches per block = warps_per_block * WARP_BATCH
- grid.x = ceil(batch_count / batches_per_block)
- **Thread dims**: `dim3(WARP_SIZE, warps_per_block, 1)`
- **NO shared memory** -- entirely register + warp shuffle based
- Each warp holds ALL elements of its row(s) in registers
- Reduction via `warp_reduce` using `WARP_SHFL_XOR`

**Sub-case 1b: Large dim_size, fast_softmax path** -- `cunn_SoftMaxForwardGmem` or `cunn_SoftMaxForwardFast`
- Block: **512 threads**
- ILP = `sizeof(float4) / sizeof(scalar_t)` (4 for float, 8 for half)
- smem = `block.x / warp_size * sizeof(accscalar_t)` (just for block reduction)
- Grid = outer_size (one block per row)

**Sub-case 1c: Large dim_size, not fast path, small reg count (<10 regs/thread)** -- `cunn_SoftMaxForwardReg`
- Block: `SoftMaxForward_getBlockSize(dim_size)` -- rounds up to multiple of warp_size, max 1024
- Elements loaded to **registers** (compile-time `reg_cnt` from 1-9)
- Three passes: load+max, reduce max, compute sum_exp, reduce sum, write output

**Sub-case 1d: Large dim_size, fits in smem** -- `cunn_SoftMaxForwardSmem`
- Condition: `dim_size * sizeof(scalar_t) < sharedMemPerBlock - reduction_smem`
- Loads ALL elements to shared memory on first pass, then rereads from smem for second pass
- smem = `dim_size * sizeof(scalar_t) + block.x / warp_size * sizeof(accscalar_t)`
- Vectorized loads (`aligned_vector<scalar_t, ILP>`)

**Sub-case 1e: Large dim_size, doesn't fit in smem** -- `cunn_SoftMaxForward`
- Three global memory passes: (1) find max, (2) compute sum of exp, (3) normalize
- Uses `ilpReduce` with vectorized loads (ILP elements per iteration)
- Uses `WriteFpropResultsVectorized` for output with alignment handling

**Case 2: inner_size > 1 (spatial softmax -- not along last dim)**
- Block: `SpatialSoftMax_getBlockSize(dim_size, inner_size)`
  - `inner_threads = min(inner_size, 1024)`
  - `dim_threads`: doubles from 1 while `inner_threads * dim_threads <= 1024` and `dim_threads <= dim_size`, only when `inner_threads <= 64` and `dim_size >= 64`
  - `block = dim3(dim_threads, inner_threads)`
- Grid: `SpatialSoftMax_getGridSize(block, max_active_blocks, outer_size, inner_size)`
  - Uses `cudaOccupancyMaxActiveBlocksPerMultiprocessor` for occupancy
  - Tiles inner over grid.y, outer over grid.x
- smem = `block.x == 1 ? 0 : block_threads * sizeof(accscalar_t)`
- Reduction via `spatialBlockReduceX` using shared memory tree reduction

### ILP Reduction Pattern (`ilpReduce`, SoftMax.cu lines 485-527)
```
ILP = sizeof(float4) / sizeof(scalar_t)  // 4 for float, 8 for half, 2 for double
```
Handles alignment:
1. If data pointer is misaligned, processes `shift` elements scalar.
2. Main loop: loads `aligned_vector<T, ILP>` per thread per iteration. Each thread strides by `blockDim.x`.
3. Epilogue: handles tail elements scalar.

### Block Reduction Patterns

**`blockReduce`** (SoftMax.cu lines 410-456): Classic shared-memory two-pass reduction.
1. All threads write to smem[threadIdx.x]
2. First warp reads all per-warp values from smem, reduces within-warp via loop over smem
3. Thread 0 reduces final per-warp values
4. Result broadcast via smem[0]

**`blockReduceWarp`** (lines 460-470): Uses `cuda_utils::BlockReduce` from `block_reduce.cuh` which does warp-shuffle first, then inter-warp via smem.

---

## 5. GENERIC REDUCTION FRAMEWORK (Reduce.cuh)

**File:** `/home/kunwar/Work/kernel_libs/pytorch/aten/src/ATen/native/cuda/Reduce.cuh`

### ReduceConfig (lines 73-217)
The central configuration object. Key fields:
- `block_width`, `block_height`, `num_threads`
- `input_mult[3]` (BLOCK_X, BLOCK_Y, CTA) -- how to map thread/block indices to input positions
- `output_mult[2]` -- how to map to output positions
- `ctas_per_output` -- for multi-block reductions
- `vectorize_input`, `output_vec_size`

### setReduceConfig (lines 1031-1178) -- THE KEY FUNCTION

This determines the entire reduction strategy. Steps:

**1. Determine fastest dimension:**
```
reduction_on_fastest_striding_dimension =
    (iter.num_reduce_dims() == iter.ndim()) ||
    (input_stride[0] < input_stride[num_reduce_dims])
```
If reducing along fastest dim: `dim0 = inputs_per_output`, `dim1 = num_outputs` (inner reduction).
Otherwise: `dim0 = num_outputs`, `dim1 = inputs_per_output` (outer reduction).

**2. Vectorization:**
- Inner reduction (reduction on fastest, dim0 >= 128, 1 reduce dim): `vectorize_input = true`, `dim0 /= input_vec_size`
- Outer reduction: `output_vec_size = get_output_vec_size()` (1, 2, or 4), `dim0 /= output_vec_size`

**3. Block dimensions** (`set_block_dimension`, lines 99-107):
```
MAX_NUM_THREADS = 512 (256 for complex double) / output_vec_size
dim0_pow2 = last_pow2(dim0), capped at max_num_threads
dim1_pow2 = last_pow2(dim1), capped at max_num_threads
block_width  = min(dim0_pow2, warp_size)  // initially
block_height = min(dim1_pow2, max_num_threads / block_width)
block_width  = min(dim0_pow2, max_num_threads / block_height)  // re-expand
num_threads  = block_width * block_height
```

**4. Split input vs output across dimensions:**
- If reducing along fastest dim: `input_mult[BLOCK_X] = split_input(block_width)` -- threads in x cooperate on reduction, need warp shuffle + smem.
- Otherwise: `output_mult[BLOCK_X] = split_output(block_width)` -- threads in x produce independent outputs.

**5. Warp splitting** (lines 1140-1148):
```
min_values_per_thread = 16 (CUDA), 128 (ROCm)
max_values_per_thread = 256
warp_split_threshold = min(block_height * 16, 256)
```
If `values_per_thread >= warp_split_threshold`: warps cooperate on input (need inter-warp smem reduction). Otherwise: each warp handles a separate output.

**6. Multi-block global reduction** (lines 1155-1176):
Triggered when `values_per_thread >= 256` and `grid <= target_grid_size`. Allocates staging buffer + semaphores for multi-block coordination via `atomicAdd` and `__threadfence`.

### Thread Reduce with Vectorization (lines 499-558)
For `vectorize_input`:
1. Handles misaligned head (scalar loads for `shift` elements)
2. Main loop: `aligned_vector<scalar_t, input_vec_size>` loads, with **input_vec_size independent accumulators** to break dependency chains
3. Handles tail
4. Combines accumulators: `value_list[0] = ops.combine(value_list[0], value_list[i])`

Default `vt0 = 4` (values per thread per output for non-vectorized), `input_vec_size = vt0`.

### Non-vectorized Thread Reduce (lines 560-631)
Also uses `vt0` independent accumulators. Each processes `output_vec_size` values in lockstep. Stride between reads = `config.step_input`.

### Block-X Reduce (lines 633-671)
If `dim_x > warpSize`: shared memory tree reduction first, then warp shuffle.
Warp shuffle: `WARP_SHFL_DOWN` for decreasing offsets (NVIDIA) or increasing offsets (ROCm).

### Block-Y Reduce (lines 673-690)
Shared memory tree reduction via `config.shared_memory_offset`.

### Global Reduce (lines 786-909)
1. Each block writes its partial result to `cta_buf` staging buffer
2. `__threadfence()` for global visibility
3. `atomicAdd(&semaphores[blockIdx.x], 1)` -- last block to finish does final reduction
4. Last block reads all partial results from staging, does block_y + block_x reduce
5. Writes final output

### Shared Memory Size (lines 183-190)
```
if (!should_block_y_reduce() && (!should_block_x_reduce() || block_width <= warpSize)):
  return 0   // pure warp shuffle, no smem needed
else:
  return element_size_bytes * num_threads * output_vec_size
```

### Kernel Launch (lines 912-933)
```
reduce_kernel<max_threads/output_vec_size, output_vec_size><<<grid, block, shared_memory>>>
```
Launch bounds: `__launch_bounds__(nt, 4)` -- targets 4 blocks per SM.

### Output Vectorization (lines 1006-1029)
`get_output_vec_size`: starts at 4, reduces by half until base address and all strides are divisible.

---

## 6. GROUP NORM KERNEL

**File:** `/home/kunwar/Work/kernel_libs/pytorch/aten/src/ATen/native/cuda/group_norm_kernel.cu`

### Forward: RowwiseMomentsCUDAKernel (lines 30-68)
- Block: `kCUDANumThreads = 256`
- Grid: one block per group (total N*G groups for N batches, G groups)
- Welford reduction over D = C/G spatial elements
- If `blockDim.x <= C10_WARP_SIZE`: uses `WarpReduce` only (no smem)
- Otherwise: `BlockReduce` with shared WelfordType array of size C10_WARP_SIZE

### ComputeFusedParams (lines 70-93)
Simple elementwise kernel: `grid = ceil(N*C / blockDim.x)`, `blockDim.x = 256`.
Computes `a[index] = rstd * gamma` and `b[index] = -a * mean + beta` for fused apply.

### Backward: `Compute1dBackwardFusedParamsCUDAKernel` (lines 95-140)
- Block: `kCUDANumThreads = 256`
- Grid: `dim3(N, G)` -- one block per (batch, group) pair
- Reduces over D = C/G elements
- Uses `WarpReduceSum` if blockDim <= warp_size, else `BlockReduceSum`
- Shared memory: 2 arrays of C10_WARP_SIZE floats

### GammaBeta Backward: `GammaBeta1dBackwardCUDAKernel2` (lines 180+)
Uses a 32x32 tiled shared memory reduction with `kReduceTileSize = 32`.
Shared memory padded to 33 (32+1) to avoid bank conflicts.

---

## 7. BATCH NORM KERNEL

**File:** `/home/kunwar/Work/kernel_libs/pytorch/aten/src/ATen/native/cuda/Normalization.cuh`

### Constants
```
MAX_BLOCK_SIZE = 512 (CUDA), 1024 (ROCm)
MAX_GRID_SIZE = 65535
ELEMENTS_PER_ITER = 4
ELEMENTS_PER_THREAD = 16
OPTIMAL_TILE_W = 32
MAX_H_BLOCK = 128
```

### `getNumThreads(nElem)` (lines 35-47)
Returns smallest power of 2 from {32, 64, 128, 256, 512} that is >= nElem.

### Statistics Collection: `batch_norm_collect_statistics_kernel` (lines 283-385)
- Grid: `dim3(C)` -- one block per channel
- Block: `dim3(tf, max(1, MAX_BLOCK_SIZE/tf))` where `tf = getNumThreads(spatial_size)`
  - e.g., spatial=1024: tf=512, block=(512,1); spatial=64: tf=64, block=(64,8)
- **Welford in thread loop**: each thread accumulates over batch*spatial elements strided by block dims
- **Two-pass warp reduction**: First `WARP_SHFL_XOR` for intra-warp Welford combine, then shared memory for inter-warp
- Shared memory: `int shared_n[2 * 2 * C10_WARP_SIZE + C10_WARP_SIZE]` + overlapping `stat_accscalar_t*`
- Max threads: C10_WARP_SIZE^2 = 1024

### Transform Input: `batch_norm_transform_input_kernel` (lines 226-263)
- Grid: `dim3(C, min(256K/C, ceil(N/tb)))` -- blocks over channels and batch
- Block: `dim3(tf, tb)` where:
  ```
  tf = max(getNumThreads(spatial/4), min(getNumThreads(spatial), 64))
  tb = max(64/tf, 1)
  ```
- Each block processes one channel across multiple batch elements
- Inner loop: `for feature in threadIdx.x..spatial step blockDim.x`
- Outer loop: `for batch in threadIdx.y + blockIdx.y*blockDim.y..N step blockDim.y*gridDim.y`

### `flexible_launch_configs` (lines 153-179)
For channels-last batch norm:
```
block_x = min(lastPow2(stride), 32)
block_y = min(lastPow2(ceil(reduction/16)), MAX_BLOCK_SIZE/block_x)
if block_x*block_y != MAX_BLOCK_SIZE:
  block_x = min(lastPow2(stride), MAX_BLOCK_SIZE/block_y)
grid_x = ceil(stride / block_x)
grid_y = min(ceil(reduction / (block_y * 16)), 128)
if coop_flag and grid_y < 8: grid_y = 1  // not worth grid reduction for small dims
```

---

## 8. WELFORD ALGORITHM IMPLEMENTATIONS

Three distinct implementations exist:

### A. SharedReduceOps.h `WelfordOps` (lines 85-149)
Used by the generic reduction framework (Reduce.cuh) and group_norm.
- `reduce(acc, data, idx)`: standard online update with integer count tracking
- `combine(a, b)`: parallel merge formula: `new_m2 = a.m2 + b.m2 + delta^2 * a.nf * nb_over_n`
- `project(acc)`: returns `(var, mean)` pair where `var = m2 / (nf - correction)`
- `warp_shfl_down`: shuffles all 4 fields (mean, m2, n, nf)

### B. layer_norm_kernel.cu `WelfordDataLN` (lines 125-244)
Custom float-only implementation optimized for layer norm:
- Uses `1.f/count` (fast reciprocal) instead of division
- On ROCm gfx942: uses `__builtin_amdgcn_rcpf` for even faster reciprocal
- RMS norm variant: only tracks `sigma2` (sum of squares), skips mean

### C. Normalization.cuh `welford_merge_element` (lines 182-193)
Used by batch norm:
- `factor = 1.0 / max(1, count + count_new)`
- Inlined manual merge, not templated on reduction op

---

## 9. BLOCK REDUCTION INFRASTRUCTURE

**File:** `/home/kunwar/Work/kernel_libs/pytorch/aten/src/ATen/native/cuda/block_reduce.cuh`

### `kCUDABlockReduceNumThreads = 512`
### `kCUDABlockReduceMaxThreads = C10_WARP_SIZE^2 = 1024`

### `WarpReduceSum` (lines 30-36)
Standard butterfly pattern: `WARP_SHFL_DOWN` for offsets warp_size/2 down to 1.

### `BlockReduceSum` (lines 77-92)
1. Warp-level reduce via `WarpReduceSum`
2. Lane 0 of each warp writes to `shared[warp_id]`
3. First warp reads all warp results, does another `WarpReduceSum`
4. Result valid only for thread 0

### Generic `BlockReduce` (lines 128-145)
Same pattern but with custom `ReduceOp` (must provide `combine` and `warp_shfl_down`).

### Supports 1D and 2D blocks via `Block1D`/`Block2D` structs that provide `Tid()` and `Warps()`.

---

## 10. INNER VS OUTER REDUCTION

### Inner Reduction (along fastest/contiguous dimension)
- `Reduce.cuh`: `reduction_on_fastest_striding_dimension = true`
- `block.x` maps to reduction dimension, requires `block_x_reduce` (warp shuffle + optional smem)
- Can vectorize input (`config.vectorize_input = true`)
- Softmax: "Regular" kernels (`cunn_SoftMaxForward`), persistent warp softmax
- LayerNorm/GroupNorm: all kernels reduce along N (feature dim), which is contiguous

### Outer Reduction (along non-contiguous dimension)
- `Reduce.cuh`: `reduction_on_fastest_striding_dimension = false`
- `block.x` maps to output dimension, each thread produces independent outputs
- Can vectorize output (`config.output_vec_size` up to 4)
- `block.y` either handles more outputs or cooperates on reduction
- Softmax: "Spatial" kernel (`cunn_SpatialSoftMaxForward`) -- strided access pattern, `data_offset + d * dim_stride`

---

## 11. KEY CONSTANTS SUMMARY

| Kernel | Block Size | Vec Size | Elements/Thread | Shared Memory |
|--------|-----------|----------|----------------|--------------|
| Elementwise (CUDA) | 128 | up to 8 | 8 (float), 16 (byte) | 0 |
| LayerNorm fwd (vec) | 128 (32x4) | 4 | N/(32*4*4) vecs/thread | 24 bytes |
| LayerNorm fwd (scalar) | 512 (moments) + 256 (apply) | 1 | N/512 | WelfordType * warp_size |
| GroupNorm fwd | 256 | 1 | D/256 | WelfordType * warp_size |
| BatchNorm stats | up to 512 (2D) | 1 | N*HW/(block_x*block_y) | ~5 * warp_size * sizeof(float) |
| Softmax persistent | 128 | 1 | WARP_ITERATIONS * WARP_BATCH | 0 |
| Softmax regular | up to 1024 | ILP (4/8/16) | dim_size / block_size | block/warp_size * sizeof(acc) |
| Softmax smem | up to 1024 | ILP | dim_size / block_size | dim_size * sizeof(T) + reduction |
| Softmax spatial | up to 1024 (2D) | 1 | dim_size / block.x | block_threads * sizeof(acc) or 0 |
| Reduce.cuh generic | up to 512 | 1-4 (in/out) | vt0=4, up to 256 | num_threads * element_size * ovec |