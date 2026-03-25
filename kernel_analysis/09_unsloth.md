# Unsloth: Memory-Bound Kernel Analysis

---

## Flashinfer Memory-Bound Kernel Analysis

### 1. RMSNorm Kernel (CUDA)

**File**: `/home/kunwar/Work/kernel_libs/flashinfer/include/flashinfer/norm.cuh`, lines 36-146

**Thread Block Configuration**:
- Grid: `dim3(batch_size)` -- one block per row/token
- Block: `dim3(32, num_warps)` -- 2D: warp_size x num_warps
- `num_warps = ceil_div(min(1024, d / vec_size), 32)` -- so total threads = `min(1024, d / vec_size)`
- For typical d=4096, fp16: vec_size=8, block_size=512, num_warps=16, threads=512

**Vector Loads**:
- `vec_size = gcd(16 / sizeof(T), d)` -- for fp16 (2 bytes), max vec_size = 8 elements = 16 bytes (128-bit load)
- Dispatched at compile time via `DISPATCH_ALIGNED_VEC_SIZE` (valid values: 1, 2, 4, 8, 16)
- Uses `vec_t<T, VEC_SIZE>` which maps to 128-bit `uint4` loads internally

**Thread-to-Data Mapping**:
- `thread_id = tx + ty * 32` (linearized from 2D block)
- Each thread handles `VEC_SIZE` contiguous elements per iteration
- Stride pattern: thread_id maps to elements `[thread_id * VEC_SIZE .. (thread_id+1) * VEC_SIZE)`
- Multiple rounds if `d > VEC_SIZE * num_threads`: `rounds = ceil_div(d, VEC_SIZE * num_threads)`

**Elements Per Thread**:
- Per round: VEC_SIZE (8 for fp16)
- Total: `rounds * VEC_SIZE` = `ceil_div(d, num_threads)` elements
- For d=4096, fp16, 512 threads: rounds=1, 8 elements/thread
- For d=8192, fp16, 1024 threads: rounds=1, 8 elements/thread

**Reduction Strategy** (two-level):
1. **Intra-warp**: `shfl_xor_sync` butterfly reduction (5 steps for warp_size=32), fully unrolled
2. **Cross-warp**: Each warp writes partial sum to `smem[ty]`, then warp 0 loads all partial sums and does another shfl_xor reduction, broadcasts result via `smem[0]`

**Memory Access Pattern**:
- Reads are coalesced: consecutive threads read consecutive VEC_SIZE-element vectors
- Thread 0 reads elements [0..7], thread 1 reads [8..15], etc. (for fp16, vec_size=8)
- Input is read TWICE (once for sum-of-squares, once for the normalization pass) -- no shared memory caching
- Weight read once in second pass, same coalesced pattern

**Shared Memory**:
- `num_warps * sizeof(float)` bytes -- only for cross-warp reduction
- For 16 warps: 64 bytes (negligible)

---

### 2. RMSNorm + FP8 Quantization Kernel (CUDA)

**File**: `/home/kunwar/Work/kernel_libs/flashinfer/include/flashinfer/norm.cuh`, lines 148-261

Identical structure to RMSNorm, with one difference in the output pass:
- Output is clamped to `[-448.0f, 448.0f]` (FP8 E4M3 range), multiplied by `1/scale`
- Uses `output_vec.cast_store()` which casts float32 to the FP8 output type during store

---

### 3. QKRMSNorm Kernel (CUDA) -- Per-Head RMSNorm

**File**: `/home/kunwar/Work/kernel_libs/flashinfer/include/flashinfer/norm.cuh`, lines 263-384

**Thread Block Configuration**:
- Grid: occupancy-based via `cudaOccupancyMaxActiveBlocksPerMultiprocessor`, capped at `num_blocks_per_sm * num_sms`
- Block: `dim3(32, 4)` -- 4 warps per block, 128 threads
- Each warp independently processes one (batch, head) job

**Thread-to-Data Mapping**:
- `worker_idx = blockIdx.x * num_warps + warp_y` -- maps to job (batch, head)
- Within a warp: 32 threads share a single head_dim reduction
- Grid-stride loop: `for (job_idx = worker_idx; job_idx < batch_size*num_heads; job_idx += num_workers)`

**Reduction Strategy**:
- **Warp-only** reduction (no cross-warp, no shared memory) -- each warp handles one head independently
- Single-level `shfl_xor_sync` butterfly

**Shared Memory**: 0 bytes

**Key difference from RMSNorm**: Designed for small d (head_dim, typically 64-128), so a single warp suffices. Multiple warps per block handle different heads in parallel.

---

### 4. FusedAddRMSNorm Kernel (CUDA)

**File**: `/home/kunwar/Work/kernel_libs/flashinfer/include/flashinfer/norm.cuh`, lines 386-514

**Thread Block Configuration**: Same as RMSNorm (1 block per row, `dim3(32, num_warps)`)

**Key Difference -- Shared Memory for Intermediate Storage**:
- `smem_size = (ceil_div(num_warps, 4) * 4 + d) * sizeof(float)` -- stores the ENTIRE `d`-dimensional intermediate result in shared memory
- For d=4096: `(16 + 4096) * 4 = 16,448 bytes`
- This avoids re-reading input and residual from global memory in the second pass

**Memory Access Pattern** (3 passes through global memory):
1. **Pass 1 (read)**: Load input + residual, compute `x = input + residual`, write `x` to smem, write updated residual back to global, accumulate sum_sq
2. **Reduction**: Same two-level shfl_xor as RMSNorm
3. **Pass 2 (write)**: Load weight from global + x from smem, compute normalized output, write to global

**Writes**: residual is UPDATED IN-PLACE (written back after adding input)

---

### 5. FusedAddRMSNorm + FP8 Quantization Kernel (CUDA)

**File**: `/home/kunwar/Work/kernel_libs/flashinfer/include/flashinfer/norm.cuh`, lines 516-648

Same as FusedAddRMSNorm but output pass includes FP8 clamping [-448, 448] and `cast_store`.

---

### 6. LayerNorm Kernel (CUDA -- TensorRT-LLM based)

**File**: `/home/kunwar/Work/kernel_libs/flashinfer/include/flashinfer/norm.cuh`, lines 743-978 (function `generalLayerNorm`)

**Thread Block Configuration**:
- Grid: `dim3(tokens)` -- one block per token
- Block: `dim3(min(hidden_dim, 1024))`, rounded up to multiple of 32
- For d=4096: block = 1024 threads

**Vector Loads**:
- Uses `packed_as<T, 2>` (vec_size=2) when hidden_dim is even and T is half/bfloat16
- This means each thread processes 2 elements at a time via half2/bfloat162 SIMD

**Thread-to-Data Mapping**:
- `n_elems = hidden_dim / num_elems_T` (halved when vectorized)
- `for (i = tidx; i < n_elems; i += blockDim.x)` -- standard grid-stride within block

**Elements Per Thread**: `ceil(hidden_dim / (blockDim.x * vec_size))` iterations, 2 elements per iteration
- For d=4096, block=1024, vec_size=2: each thread processes 2 elements

**Reduction Strategy** (TensorRT-LLM utilities):
- `blockReduceSum`: warp-level shfl_xor (5 steps) + shared memory (`static __shared__ T shared[32]`) + second warp-level reduction
- Mean computed by thread 0, broadcast via `__shared__ float s_mean, s_variance`
- Two-pass (default) or one-pass (USE_DIFF_OF_SQUARES) variance computation

**Shared Memory**:
- When `USE_SHMEM=true`: `hidden_dim * sizeof(T)` bytes (caches entire input row)
- Falls back to `USE_SHMEM=false` (re-reads from global) if smem > 48KB
- Plus static shared: `float shared[32]` for warp reduction + `s_mean` + `s_variance`
- For d=4096, bf16: 8192 bytes (fits in 48KB)

---

### 7. Activation Kernel (act_and_mul -- SiLU/GeLU/GeLUTanh) (CUDA)

**File**: `/home/kunwar/Work/kernel_libs/flashinfer/include/flashinfer/activation.cuh`, lines 28-64

**Thread Block Configuration**:
- Grid: `dim3(num_tokens)` -- one block per token
- Block: `min(d / vec_size, 1024)` threads (1D)
- For d=4096, fp16: vec_size=8, block=512 threads

**Vector Loads**:
- `vec_size = 16 / sizeof(T)` -- hardcoded 128-bit loads (8 for fp16, 4 for fp32)
- Uses `vec_t<float, vec_size>` with `cast_load` -- loads T, casts to float in registers

**Thread-to-Data Mapping**:
- 1D grid-stride loop: `for (idx = thread_idx; idx < d / vec_size; idx += stride)`
- Each thread processes one vector of `vec_size` elements per iteration
- Reads from TWO halves: `input[offset + idx*vec_size]` (activation half) and `input[offset + d + idx*vec_size]` (gate half)
- `#pragma unroll 1` on outer loop -- explicitly prevents unrolling for register pressure

**Elements Per Thread**: `ceil(d / (vec_size * blockDim.x)) * vec_size` total elements
- For d=4096, fp16, 512 threads: exactly 8 elements/thread (1 iteration)

**Reduction Strategy**: None -- pure elementwise

**Memory Access Pattern**:
- Two coalesced reads (activation + gate) from input tensor of shape `[tokens, 2*d]`
- One coalesced write to output of shape `[tokens, d]`
- Tail handling: separate scalar loop for `d % (stride * vec_size)` remaining elements

**Shared Memory**: 0 bytes

---

### 8. PackBits Kernel (CUDA)

**File**: `/home/kunwar/Work/kernel_libs/flashinfer/include/flashinfer/quantization.cuh`, lines 39-63

**Thread Block Configuration**:
- Block: 256 threads
- Grid: `ceil_div(num_elements, 256 * 8)`

**Vector Loads**: Uses CUB `BlockLoad<bool, 256, 8, BLOCK_LOAD_VECTORIZE>` -- each thread loads 8 bools, with vectorized access pattern

**Elements Per Thread**: 8 bools, packed into 1 uint8_t output

**Shared Memory**: `BlockLoad::TempStorage` (CUB internal, typically ~2KB for this config)

---

### 9. Triton RMSNorm Kernel

**File**: `/home/kunwar/Work/kernel_libs/flashinfer/flashinfer/triton/kernels/norm.py`, lines 7-77

**Launch config** (from `/home/kunwar/Work/kernel_libs/flashinfer/flashinfer/triton/norm.py`, lines 23-46):
- Grid: `(b,)` -- one program per row
- `BLOCK_SIZE = triton.next_power_of_2(n)` -- rounds hidden_size up to power of 2
- `num_warps = max(8, min(32, BLOCK_SIZE // 256))`
  - For n=4096: BLOCK_SIZE=4096, num_warps=16
  - For n=8192: BLOCK_SIZE=8192, num_warps=32

**Thread-to-Data Mapping**:
- `offsets = off + tl.arange(0, BLOCK_SIZE)` with mask `offsets < n`
- Two loops over `range(0, n, BLOCK_SIZE)` -- if n <= BLOCK_SIZE, single iteration (common case since BLOCK_SIZE is rounded up)
- Each program processes entire row in BLOCK_SIZE-element chunks

**Reduction**: `tl.sum(square_sum)` (Triton's built-in block-level sum reduction)

**Memory Reads**: Two passes through the row (once for norm computation, once for output), same as CUDA version

**For fused add residual** (`rms_norm_add_residual`):
- `num_warps = min(32, triton.cdiv(BLOCK_SIZE, 32))` -- different formula
- Residual is written back, then re-read in second pass

---

### 10. Triton SiLU-and-Mul Kernel

**File**: `/home/kunwar/Work/kernel_libs/flashinfer/flashinfer/triton/kernels/activation.py`, lines 7-64

**Launch config** (from `/home/kunwar/Work/kernel_libs/flashinfer/flashinfer/triton/activation.py`, lines 41-55):
- Grid: `(b, triton.cdiv(d, BLOCK_SIZE))` -- 2D: rows x column tiles
- `BLOCK_SIZE = 1024` (hardcoded constant)
- num_warps not explicitly set (Triton default, typically 4)

**Thread-to-Data Mapping**:
- 2D grid: `i = program_id(0)` (row), `j = program_id(1)` (column tile)
- `offsets = j * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)`, mask `offsets < d`
- Each program loads 1024 elements from each of the two halves (a and b)

**Key detail**: Unlike the CUDA version which is 1D (one block per row), the Triton version tiles across columns, enabling multiple CTAs per row for large d.

---

### 11. MXFP8 Quantization Kernels (CuTe-DSL)

**File**: `/home/kunwar/Work/kernel_libs/flashinfer/flashinfer/quantization/kernels/mxfp8_quantize.py`

**Constants** (from `quantization_cute_dsl_utils.py`, lines 35-47):
- `SF_VEC_SIZE = 32` -- each scale factor covers 32 elements
- `ELTS_PER_THREAD = 8` -- each thread handles 8 fp16 elements (128 bits)
- `THREADS_PER_SF = 4` -- 4 threads cooperate per scale factor block (32/8)
- `SF_BLOCKS_PER_WARP = 8` -- 32/4 = 8 SF blocks per warp

#### Linear Layout Kernel (`MXFP8QuantizeLinearKernel`):
- **Block**: 16 warps x 32 = 512 threads
- **Grid**: `min(total_sf_blocks / SF_BLOCKS_PER_TB, num_SMs * 4)` where `SF_BLOCKS_PER_TB = 16 * 8 = 128`
- **Occupancy target**: `_BLOCKS_PER_SM = 4`
- **Vector loads**: `ld_global_v4_u32` -- 4x 32-bit loads = 128 bits per thread (8 fp16 elements)
- **Reduction**: 4-thread max reduction within each SF block via `reduce_max_4threads` (shfl-based)
- **Thread-to-Data**: SF-block based iteration with grid-stride loop. Each group of 4 threads processes 32 contiguous elements.
- **Store**: `st_global_u64` -- 64-bit store of 8 packed FP8 values per thread

#### Swizzled Layout Kernel (`MXFP8QuantizeSwizzledKernel`):
- **Block**: Dynamic warps via `_compute_optimal_warps_for_k(K)` (range [4, 32] warps)
  - Ensures `(WARPS * 8) % (K/32) == 0` for 100% thread utilization
- **Dual-path**:
  - **Small K (K <= 4096)**: Multi-row processing. `rows_per_block = col_units / sf_blocks_per_row`. Thread mapping: `row_in_block = tidx / threads_per_row`, `sf_col_idx = (tidx % threads_per_row) / THREADS_PER_SF`
  - **Large K**: Single row per block iteration, column loop

---

### 12. MXFP4 Quantization Kernel (CuTe-DSL)

**File**: `/home/kunwar/Work/kernel_libs/flashinfer/flashinfer/quantization/kernels/mxfp4_quantize.py`

**Constants**:
- `MXFP4_SF_VEC_SIZE = 32` -- 32 elements per SF block
- Each thread processes one FULL SF block (32 elements) -- vs MXFP8 where 4 threads cooperate

**Thread Configuration**:
- Dynamic: `_compute_optimal_threads_for_k(K)` (range [128, 512])
- Finds largest multiple of `threads_per_row = K/32` that fits in [128, 512]

**Dual-path** (same structure as MXFP8 swizzled):
- Small K: multi-row, `rows_per_block = num_threads / threads_per_row`
- Large K: single row with column loop

**Memory Access**: Each thread does 4x 128-bit loads (32 fp16 elements = 512 bits total input), produces 16 bytes (32 FP4 values) stored via 2x `st_global_u64`

---

### 13. CuTe-DSL RMSNorm Kernel

**File**: `/home/kunwar/Work/kernel_libs/flashinfer/flashinfer/norm/kernels/rmsnorm.py`

**Thread Configuration** (lines 154-170):
```
H <= 64:    threads_per_row = 8
H <= 128:   threads_per_row = 16
H <= 3072:  threads_per_row = 32
H <= 6144:  threads_per_row = 64
H <= 16384: threads_per_row = 128
H > 16384:  threads_per_row = 256

num_threads = 128 if H <= 16384 else 256
rows_per_block = num_threads / threads_per_row
```

So for typical LLM hidden sizes:
- d=4096: 32 threads/row, 128 total, 4 rows/block
- d=8192: 128 threads/row, 128 total, 1 row/block
- d=16384: 128 threads/row, 128 total, 1 row/block

**Cluster support** (SM 90+): Splits H across cluster_n CTAs for very large hidden sizes. `cluster_n` chosen so `H_per_cta` fits in shared memory.

**Vectorization**: `vec_size = min(H_per_cta & (-H_per_cta), COPY_BITS / 8 / elem_bytes)` where `COPY_BITS = 128` (128-bit copies). For fp16: max vec_size = 8.

**Shared Memory**: `tile_bytes + rows_per_block * warps_per_row * 4` -- stores the entire tile (input data) + reduction scratch. Uses async copy (`cp.async`) when tile fits in half the available shared memory.

---

### Summary Table of Key Parameters

| Kernel | Block Size | Vec Size (fp16) | Elements/Thread | Reduction | Shared Mem | Passes |
|--------|-----------|-----------------|-----------------|-----------|------------|--------|
| RMSNorm (CUDA) | min(1024, d/vec) | 8 (128-bit) | ceil(d/threads)*vec | shfl_xor + smem cross-warp | num_warps*4B | 2 (input read twice) |
| RMSNormQuant (CUDA) | same | 8 | same | same | same | 2 |
| QKRMSNorm (CUDA) | 128 (4 warps) | 8 | ceil(head_dim/32)*vec | shfl_xor only (warp-level) | 0 | 2 |
| FusedAddRMSNorm (CUDA) | min(1024, d/vec) | 8 | same | same as RMSNorm | (pad+d)*4B | 2 (but input read once, smem cached) |
| LayerNorm (TRT-LLM) | min(d, 1024), aligned to 32 | 2 (half2 SIMD) | ceil(d/(block*2))*2 | blockReduceSum (shfl + smem[32]) | d*sizeof(T) or 0 | 2 or 3 |
| act_and_mul (CUDA) | min(d/vec, 1024) | 8 (128-bit) | ceil(d/(block*vec))*vec | None | 0 | 1 |
| Triton RMSNorm | num_warps * 32 | auto (Triton) | BLOCK_SIZE / (num_warps*32) | tl.sum | auto | 2 |
| Triton SiLU-and-Mul | default (4 warps) | auto | 1024 per program | None | 0 | 1 |
| MXFP8 Quant Linear | 512 (16 warps) | 128-bit (v4_u32) | 8 per thread | 4-thread max (shfl) | 0 | 1 |
| MXFP8 Quant Swizzled | dynamic (4-32 warps) | 128-bit | 8 per thread | 4-thread max | 0 | 1 |
| MXFP4 Quant | dynamic (128-512 threads) | 4x128-bit per thread | 32 per thread | per-thread (no cross-thread) | 0 | 1 |
| PackBits | 256 | CUB vectorized | 8 bools | None | CUB TempStorage | 1 |