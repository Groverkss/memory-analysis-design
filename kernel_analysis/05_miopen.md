# MIOpen (AMD ROCm): Memory-Bound Kernel Analysis

---

# MIOpen Memory-Bound Kernel Analysis for AMD GPUs

## 1. LAYERNORM (LayerNorm, AddLayerNorm, T5LayerNorm)

### Files
- **Kernel**: `/home/kunwar/Work/kernel_libs/MIOpen/src/kernels/MIOpenLayerNorm.cpp`
- **Solver**: `/home/kunwar/Work/kernel_libs/MIOpen/src/solver/layernorm/forward_layernorm.cpp`
- **T5 Solver**: `/home/kunwar/Work/kernel_libs/MIOpen/src/solver/layernorm/forward_t5layernorm.cpp`
- **Utils**: `/home/kunwar/Work/kernel_libs/MIOpen/src/include/miopen/layernorm/utils.hpp`

### Workgroup Configuration
- **Workgroup size**: **256 threads** (hardcoded `#define LOCAL_SIZE 256`, solver line 35)
- **Grid**: `outer_size * 256` in x dimension only. One workgroup per "outer" instance (e.g., for `{N,C,L}` normalized over `{L}`, outer_size = N*C).
- **Occupancy target**: At least 4 WGs per CU (`get_reqd_work_item_cnt` = `LOCAL_SIZE * MaxComputeUnits * 4`, utils.hpp line 58)

### Thread-to-Data Mapping
- Each workgroup handles one "row" of the normalization dimension (line 62-63: `gid = blockIdx.x`, `lid = threadIdx.x`).
- Thread `lid` processes elements `i = lid, lid + LOCAL_SIZE, lid + 2*LOCAL_SIZE, ...` up to `inner_size` (stride loop at line 71).
- This is a **strided access pattern** -- consecutive threads access consecutive elements within the row, giving **perfect coalescing** for the 64-wide wavefront on AMD.

### Elements Per Thread
- `ceil(inner_size / 256)` elements per thread. For a typical hidden_size of 768, that's 3 elements/thread. For 4096, that's 16 elements/thread.

### Vector Loads
- **None**. All loads are scalar: `x[x_idx]` (line 75). No explicit vectorization via `float4` or buffer_load intrinsics.

### Reduction Strategy
- **LDS-based tree reduction** (no wavefront-level DPP/shuffle):
  - Each thread accumulates partial sum into registers (lines 71-78 for mean+var).
  - Results written to `__shared__ FLOAT_ACCUM ltmp1[LOCAL_SIZE]` and `ltmp2[LOCAL_SIZE]` (lines 67-68).
  - Classic power-of-two tree reduction: `for(i = LOCAL_SIZE >> 1; i > 0; i >>= 1)` with `__syncthreads()` at each step (lines 83-91).
  - For 256 threads, this is 8 reduction steps with 8 barriers.
  - **No wavefront-level optimization** -- even the final steps within a single wavefront still use LDS + syncthreads (this is suboptimal on AMD where DPP could reduce the last 6 steps to wavefront intrinsics).

### LDS Usage
- **LayerNorm/GroupNorm**: 2 arrays of `LOCAL_SIZE` (256) FLOAT_ACCUM = **2 * 256 * 4 = 2048 bytes** (fp32 accum).
- **T5LayerNorm**: 1 array = **1024 bytes**.
- Checked against `TargetProperties::GetMaxLocalMemorySize()` (solver line 54).
- **No bank conflict avoidance** -- sequential indexing into `ltmp[lid]` is naturally conflict-free since consecutive threads access consecutive addresses.

### Memory Access Pattern
- **Two-pass**: First pass reads all `x` for reduction (lines 71-78). Second pass reads `x` again for normalization output (lines 105-119).
- **Re-reads input from global memory** -- no register caching of input between reduction and normalization passes.
- Contiguous-only path -- kernel assumes packed tensors (`IsAllPacked()` check in solver).

### AMD-Specific: Wavefront Size Impact
- 256 threads = **4 wavefronts of 64 lanes** on GFX8/GFX9, or **8 wavefronts of 32 lanes** on GFX10+/GFX11+/GFX12+.
- The tree reduction using `__syncthreads()` is **not** wavefront-aware. On AMD GFX9 with 64-wide wavefronts, the last 6 steps of the tree reduction could be replaced with a single wavefront shuffle reduction + 2-step inter-wavefront reduction via LDS. Instead, all 8 steps use full barriers.

---

## 2. GROUPNORM

### Files
- **Kernel**: `/home/kunwar/Work/kernel_libs/MIOpen/src/kernels/MIOpenGroupNorm.cpp`
- **Solver**: `/home/kunwar/Work/kernel_libs/MIOpen/src/solver/groupnorm/forward_groupnorm.cpp`

### Workgroup Configuration
- **Workgroup size**: **1024 threads** (hardcoded `#define LOCAL_SIZE 1024`, solver line 35)
- **Grid**: `(N * num_groups) * 1024` in x.
- **Applicability guard** (solver line 63-64): `N * num_groups >= 32` and `C / num_groups < 64`. This means the solver only applies when each group has **few channels** (< 64) and there are enough groups to saturate the GPU.

### Thread-to-Data Mapping
- Identical pattern to LayerNorm but with `inner_size = numel_per_channel * num_channels / num_groups` (line 67).
- Each workgroup processes one group within one batch element.
- Stride loop: `for(i = lid; i < inner_size; i += LOCAL_SIZE)`.

### Elements Per Thread
- `ceil(inner_size / 1024)`. GroupNorm inner_size = (C/G) * H * W. With C/G < 64 and, say, 32x32 spatial, inner_size = 64*1024 = 65536, so ~64 elements/thread.

### LDS Usage
- 2 arrays of 1024 FLOAT_ACCUM = **2 * 1024 * 4 = 8192 bytes**. Checked against max local memory (solver line 61).

### Reduction Strategy
- Same as LayerNorm: LDS tree reduction, no wavefront intrinsics.

### Key Difference from LayerNorm
- 4x larger workgroup (1024 vs 256), reflecting that GroupNorm's inner dimension is typically larger (includes spatial dims).

---

## 3. SOFTMAX (OpenCL Legacy Kernel)

### Files
- **Kernel**: `/home/kunwar/Work/kernel_libs/MIOpen/src/kernels/MIOpenSoftmax.cl`
- **Solver**: `/home/kunwar/Work/kernel_libs/MIOpen/src/solver/softmax/softmax.cpp`

### Workgroup Configuration
- **Workgroup size**: **256 threads** (hardcoded `out_vld = {256, 1, 1}`, solver line 90).
- **Two dispatch strategies** based on `vector_size` (number of channels/elements to reduce):

**CSR-Vector (NUM_BATCH == 1)**: When `vector_size >= 256`:
- Grid: `min(grid_size, 64 * 40 * 8) * 256` (solver line 102). The magic number `64 * 40 * 8 = 20480` is `~CUs * wavefronts_per_CU * some_factor` for a large AMD GPU.
- One workgroup per spatial location; entire workgroup cooperates on channel reduction.
- Three reads of global memory (max, exp+sum, normalize -- lines 239-410 of the kernel).

**CSR-Stream (NUM_BATCH > 1)**: When `vector_size < 256`:
- Multiple spatial dims packed into one workgroup.
- `num_batch = nextPow2(256 / vector_size)` (solver line 88) -- e.g., if vector_size=31, num_batch=8, batch_size=32.
- `u_batch_size = nextPow2(vector_size / batch_size)` for thread iteration count.

### Thread-to-Data Mapping (CSR-Vector)
- Thread `lid` processes channels `i = lid, lid+256, lid+512, ...` (line 242: `for(i = lid; i < vector_size; i += get_local_size(0))`).
- Coalesced access for mode=channel (stride-1 across channels).

### Thread-to-Data Mapping (CSR-Stream)
- `batch_lid = lid & (BATCH_SIZE - 1)` -- thread's position within its batch.
- `batch = lid / BATCH_SIZE` -- which spatial dim this thread works on.
- Each thread stores intermediate values in private registers (`_FLOAT channel[U_BATCH_SIZE]` at the kernel level).

### Reduction Strategy
- **LDS tree reduction** (line 218: `local _FLOAT l_helper[256]`):
  - For max: `l_helper[lid] = max(l_helper[lid], l_helper[lid + i])` (line 274)
  - For sum: `l_helper[lid] += l_helper[lid + i]` (line 337-343)
  - Same `for(i = get_local_size(0) >> 1; i > 0; i >>= 1)` pattern.
- No wavefront intrinsics in this OpenCL path.

### LDS Usage
- `256 * sizeof(_FLOAT)` = 1024 bytes (fp32) or 512 bytes (fp16).

### Memory Access Pattern
- **Three-pass** for accurate mode: (1) find max, (2) compute exp and sum, (3) normalize.
- Comment at line 373-375 explicitly states: "Subtracting max again because we do not write the output of value-max to DRAM above. Doing a subtraction again is much faster than writing uncoalesced to DRAM."

---

## 4. SOFTMAX ATTENTION (HIP Kernel, AMD-Optimized)

### Files
- **Kernel**: `/home/kunwar/Work/kernel_libs/MIOpen/src/kernels/MIOpenSoftmaxAttn.cpp`
- **Solver**: `/home/kunwar/Work/kernel_libs/MIOpen/src/solver/softmax/attn_softmax.cpp`
- **Warp size header**: `/home/kunwar/Work/kernel_libs/MIOpen/src/kernels/miopen_warp_size.hpp`

### Wavefront Size Awareness
The `miopen_warp_size.hpp` file (lines 28-32) is critical:
```cpp
#if defined(__GFX8__) || defined(__GFX9__)
#define MIOPEN_WARP_SIZE 64
#else
#define MIOPEN_WARP_SIZE 32
#endif
```
GFX8/GFX9 (MI100, MI200, MI250, MI300) use wavefront 64. GFX10+ (RDNA) use wavefront 32.

### Workgroup Configuration
- **Dynamic workgroup size**: `local_threads = clamp(nextPow2(seq_len), warpSize, 256)` (solver line 106).
- So: seq_len=16 -> local_threads=64 (one wavefront on GFX9); seq_len=128 -> local_threads=128; seq_len=300 -> local_threads=256.
- Max workgroup is 256, using `__launch_bounds__(THREADS)`.

### Three Kernel Variants
Selected based on seq_len vs warpSize vs local_threads (solver lines 117-119):

1. **SoftMaxWarp** (`seq_len <= warpSize`, i.e., seq_len <= 64 on GFX9):
   - **Entire softmax computed within a single wavefront** -- no LDS needed for reduction.
   - Multiple rows packed per workgroup: `NumWarps = THREADS / MIOPEN_WARP_SIZE`.
   - Grid: `global_threads = Ceil(nhs * local_threads, local_threads / warpSize)` (solver line 123).
   - Loop at line 231: `for(gid = blockIdx.x * NumWarps + warpId; gid < nhs; gid += gridDim.x * NumWarps)`.

2. **SoftMaxBlock** (`warpSize < seq_len <= local_threads`, i.e., 64 < seq_len <= 256 on GFX9):
   - One row per workgroup. Block-level reduction across wavefronts.

3. **SoftMaxCommon** (`seq_len > local_threads`, i.e., seq_len > 256):
   - One row per workgroup, threads loop over elements with stride `blockDim.x`.

### AMD-Specific Reduction: `reductionFullWarp`
This is the most AMD-specific code in MIOpen (kernel lines 80-134):

**For swizzle sizes < 64** (within a 32-lane half-wavefront):
- Uses `__hip_ds_swizzlef_N<swizzle_op>` -- AMD's DS_SWIZZLE instruction for butterfly reduction.
- Constructs the swizzle opcode from xor_mask, or_mask, and_mask fields (lines 109-127).
- This is a **single-cycle cross-lane operation within 32 lanes** without LDS.

**For swizzle sizes >= 64** (crossing 32-lane boundary within a 64-wide wavefront):
- Uses `__builtin_amdgcn_ds_bpermute` (line 102) -- AMD's DS_BPERMUTE instruction.
- This uses LDS hardware for cross-lane data movement but is much faster than explicit LDS read/write.
- `idx = laneId ^ (SWIZZLE_SIZE >> 1)` gives butterfly pattern.

**Block-level reduction** (`reductionBlock`, line 136-160):
```cpp
template <uint32_t NumWarps, typename Op>
__forceinline__ __device__ float
reductionBlock(float local_val, Op op, uint32_t lid, uint32_t laneId, uint32_t warpId)
{
    static_assert(NumWarps <= MIOPEN_WARP_SIZE);
    __shared__ float reduction_tmp[NumWarps];
    float reduced_val = reductionFullWarp<MIOPEN_WARP_SIZE>(local_val, laneId, op);
    if(laneId == 0) reduction_tmp[warpId] = reduced_val;
    __syncthreads();
    if(lid < NumWarps) {
        reduced_val = reductionFullWarp<NumWarps>(reduction_tmp[lid], laneId, op);
        if(lid == 0) reduction_tmp[0] = reduced_val;
    }
    __syncthreads();
    return reduction_tmp[0];
}
```
- Phase 1: Intra-wavefront reduction via swizzle/bpermute (all 64 lanes down to 1 value per wavefront).
- Phase 2: One value per wavefront written to LDS (only `NumWarps` entries, max 4 for 256 threads on GFX9).
- Phase 3: First wavefront reduces the `NumWarps` values using the same swizzle machinery.
- **Total LDS**: only `NumWarps * 4 bytes` = 16 bytes for a 256-thread workgroup on GFX9.

### SoftMaxWarp: Pure Wavefront Reduction (No LDS)
For seq_len <= 64 (one wavefront handles one row):
- Line 242: `float r_max = reductionFullWarp<MIOPEN_WARP_SIZE>(local_val, laneId, fmaxf_op)` -- reduces 64 values to 1 using only cross-lane ops.
- Line 246: `float r_sum = 1.0f / reductionFullWarp<MIOPEN_WARP_SIZE>(local_val, laneId, plus_op)`.
- **Zero LDS usage** for the softmax computation itself (only `NumWarps` floats for Amax if needed).

### Memory Access Pattern
- Single-pass for warp/block variants: read once, compute max in registers, recompute exp inline.
- SoftMaxCommon: Three passes (max reduction, sum reduction, normalize) but same memory access pattern as the generic approach.

---

## 5. BATCHNORM SPATIAL (Training)

### Files
- **Kernel**: `/home/kunwar/Work/kernel_libs/MIOpen/src/kernels/MIOpenBatchNormFwdTrainSpatial.cl`
- **Solver**: `/home/kunwar/Work/kernel_libs/MIOpen/src/solver/batchnorm/forward_spatial.cpp`
- **Reduction functions**: `/home/kunwar/Work/kernel_libs/MIOpen/src/kernels/reduction_functions.h`
- **Config heuristics**: `/home/kunwar/Work/kernel_libs/MIOpen/src/include/miopen/batchnorm/common_spatial.hpp`

### Workgroup Configuration
BatchNorm has the most complex configuration with **5 variants** and **tuning search**.

**Variant 0/1/3 (Spatial Single -- NCHW)**:
- Workgroup size: **1024** (or 256 if `in_cstride < 256 && n < 256`, solver lines 237-244)
- Grid: `c * xlocalsize` in x -- one workgroup per channel.
- 1D workgroup.

**Variant 2 (Spatial Multiple -- NCHW)**:
- Up to **3D workgroups**: `(xlocalsize, ylocalsize, zlocalsize)`.
- NCHW default: `xlocalsize=1`, `ylocalsize=1024` (common_spatial.hpp line 103).
- If `in_cstride` < ylocalsize: `ylocalsize = max(64, nextPow2(in_cstride/vectorsize))`.
- Grid split across channels (x), spatial dims (y), and batch (z).

**Variant 2 (Spatial Multiple -- NHWC)**:
- 2D/3D workgroups with xlocalsize in [2..64], ylocalsize in [8..32].
- `xlocalsize` adjusted by channel count: `min(nextPow2(c/vectorsize), limit)`.
- `xlocalsize_limit` = 16 (fp32 vectorized), 32 (fp16 vectorized), 64 (scalar) (common_spatial.hpp line 55).
- Max total threads per WG: `1024 / vectorsize` (line 57).
- **Occupancy-driven**: decreases max_localsize until `nworkgroups >= GetMinWorkgroups()` (line 62), where MinWorkgroups = ~80% of CUs.

### Vector Loads
- **Vectorized**: sizes 1, 2, 4, 8 (via `MIO_BN_VEC_SIZE`, solver line 329).
- NHWC heuristics select vector size by channel count (common_spatial.hpp lines 480-547):
  - c <= 64: vectorsize=2
  - c == 256, in_cstride >= 1024: vectorsize=8
  - c == 512, in_cstride >= 256: vectorsize=8
  - c >= 1024: vectorsize=8 (typically)
- NCHW uses vectorization on the HW dimension: `in_cstride % vectorsize == 0` required.
- Types used: `_FLOAT2`, `_FLOAT4`, `_FLOAT8` via OpenCL vector types.

### AMD-Specific: DPP Reduction (GCN Path)
The `reduction_functions.h` file has the most AMD-specific code (lines 94-191).

**Enabled on**: GFX8, GFX9 (disabled on GFX10x, GFX11x, GFX12x via `MIOPEN_USE_AMDGCN` flag).

**DPP (Data Parallel Primitives) instructions** (lines 96-111):
```asm
v_add_f32 %0 %0 %0 row_shr:1 bound_ctrl:0   ; shift right by 1 within row of 16
v_add_f32 %0 %0 %0 row_shr:2 bound_ctrl:0   ; shift right by 2
v_add_f32 %0 %0 %0 row_shr:4 bank_mask:0xe  ; shift right by 4
v_add_f32 %0 %0 %0 row_shr:8 bank_mask:0xc  ; shift right by 8
v_add_f32 %0 %0 %0 row_bcast:15 row_mask:0xa ; broadcast lane 15 across rows
v_add_f32 %0 %0 %0 row_bcast:31 row_mask:0xc ; broadcast lane 31 across rows
```
This is a **6-step wavefront-level reduction** for 64 lanes:
- Steps 1-4: reduce within each 16-lane row (DPP row_shr with progressive offsets)
- Step 5: `row_bcast:15` -- broadcast result from row 0 to row 1 (reducing 4 rows to 2)
- Step 6: `row_bcast:31` -- broadcast result from rows 0-1 to rows 2-3 (final reduction)

**Interleaved DPP** (lines 114-137): Reduces two values simultaneously (mean AND variance), interleaving the DPP instructions for ILP.

**Triple DPP** (lines 140-163): Three simultaneous reductions (used in backward pass for dx, dmean, dvar).

**GCN cross-wavefront reduction** (`gcn_reduce2`, lines 165-189):
```c
unsigned int ldsidx = lid >> 6;  // wavefront ID = lid / 64
dpp_interleaved_reduction(x, y);  // reduce within wavefront
if((lid % 64) == 63) {           // last lane of each wavefront
    lcl_data_x[ldsidx] = *x;    // store to LDS
    lcl_data_y[ldsidx] = *y;
}
barrier(CLK_LOCAL_MEM_FENCE);
// Sequential accumulation of wavefront results
for(unsigned int i = 0; i < MIO_BN_LDSGCN_SIZE; i++) {
    *x += lcl_data_x[i];
}
```
- `MIO_BN_LDSGCN_SIZE = xlocalsize / 64` (solver line 243). For 1024 threads: 16 wavefronts, 16 LDS entries.
- Only lane 63 of each wavefront writes to LDS (the last lane has the complete reduction result from DPP).

**Non-GCN path** (`lds_reduce2`): Classic LDS tree reduction for `MIO_BN_LDS_SIZE` entries, identical to LayerNorm.

### LDS Usage
- GCN path: `MIO_BN_LDSGCN_SIZE * sizeof(float) * 2` = for 1024 threads: `16 * 4 * 2 = 128 bytes` (dramatically less than the non-GCN path).
- Non-GCN path: `MIO_BN_LDS_SIZE * sizeof(float) * 2` = for 1024 threads: `1024 * 4 * 2 = 8192 bytes`.

### Thread-to-Data Mapping (Variant 0)
- Compile-time macros define the data decomposition:
  - `MIO_BN_SEGMENT = min(MIO_BN_GRP0/MIO_BN_HW * MIO_BN_HW, MIO_BN_NHW)` (line 58)
  - `MIO_BN_NLOOP = ceil(NHW / SEGMENT)` (line 59)
  - `MIO_BN_SEGIHW = SEGMENT / HW` (line 60)
- Each thread loads `MIO_BN_NLOOP` values into a private register array `_FLOAT batchvalues[MIO_BN_NLOOP]` (line 91).
- Batch index: `nid = n * MIO_BN_SEGIHW + lidihw` where `lidihw = lid / MIO_BN_HW` (line 101).
- **Key pattern**: Thread locality maps to `(batch_sample, spatial_position)` pairs, where `lid % MIO_BN_HW` determines spatial position and `lid / MIO_BN_HW` determines which batch sample.

### Occupancy/CU Targeting
- The NHWC spatial multiple path (common_spatial.hpp line 62) has explicit CU-awareness:
  ```cpp
  while(nworkgroups < problem.GetMinWorkgroups() && max_localsize >= xlocalsize_limit)
  ```
  `GetMinWorkgroups()` returns ~80% of available CUs, ensuring sufficient workgroups to saturate the GPU.

---

## 6. BATCHNORM INFERENCE (Spatial)

### Files
- **Kernel**: `/home/kunwar/Work/kernel_libs/MIOpen/src/kernels/MIOpenBatchNormFwdInferSpatial.cl`
- **Solver**: `/home/kunwar/Work/kernel_libs/MIOpen/src/solver/batchnorm/forward_inference.cpp`

### Workgroup Configuration
- 2D workgroup: `(MIO_BN_GRP0, MIO_BN_GRP1, 1)`.
- **Elementwise operation** -- no reduction needed. Just applies `(x - mean) * invVariance * scale + bias`.

### Vector Loads
- Uses OpenCL vector types for vectorized loads:
  - `_FLOAT_PREC_C` (vectorized precision type) for reading mean/variance/scale/bias.
  - `_FLOAT_LS` for reading input values (line 78: `value = *((const __global _FLOAT_LS*)(in + index))`).
  - `VEC_SIZE_X` and `VEC_SIZE_Y` control vectorization.
- Bounds check: `if(xgid * VEC_SIZE_X >= c || ygid * VEC_SIZE_Y >= hw) return` (line 60).

### Memory Access Pattern
- Perfectly coalesced: 2D grid maps to (channel, spatial) dimensions. Consecutive threads access consecutive channels (x dimension) or consecutive spatial locations (y dimension).
- Loops over batch dimension in the innermost loop: `for(int n = 0; n < batchSize; n++)` (line 75).

---

## 7. REDUCE SUM / REDUCE CALCULATION / REDUCE EXTREME

### Files
- **ReduceSum kernel**: `/home/kunwar/Work/kernel_libs/MIOpen/src/kernels/MIOpenReduceSum.cpp`
- **ReduceCalculation kernel**: `/home/kunwar/Work/kernel_libs/MIOpen/src/kernels/MIOpenReduceCalculation.cpp`
- **ReduceExtreme kernel**: `/home/kunwar/Work/kernel_libs/MIOpen/src/kernels/MIOpenReduceExtreme.cpp`
- **Block reduce**: `/home/kunwar/Work/kernel_libs/MIOpen/src/kernels/block_reduce.hpp`
- **Warp reduce**: `/home/kunwar/Work/kernel_libs/MIOpen/src/kernels/warp_reduce.hpp`
- **Sum solver**: `/home/kunwar/Work/kernel_libs/MIOpen/src/solver/reduce/forward_sum.cpp`

### Workgroup Configuration
- All use `LOCAL_SIZE = 256`.
- **Two-kernel strategy** for parallelism:
  1. `CalculationParallelFwdContiguous`: grid = `AlignUp(parallelism_size * output_numel, 256)` -- each thread handles one output element, multiple threads cooperate on the reduce dimension.
  2. `CalculationFwdContiguous`: grid = `AlignUp(output_numel, 256)` -- final reduction, one thread per output.
- **Parallelism size** (utils.hpp lines 69-78): `parallelism_size = 1` initially, doubled until `parallelism_size * inner_size >= reqd_work_item_cnt` or `parallelism_size >= sqrt(outer_size)`. This ensures enough work items to saturate the GPU.

### ReduceSum: Wavefront-Aware Block Reduce
`block_reduce.hpp` (lines 69-83) -- the most wavefront-aware reduce code:
```cpp
template <BinaryOp_t Op, uint64_t reduce_size, ReduceThreadDim thread_dim>
__device__ FLOAT_ACCUM block_reduce(FLOAT_ACCUM val) {
    if(reduce_size == warpSize)
        return warp_reduce<Op>(val);  // single wavefront, no LDS
    if(warpSize == 32)
        return block_reduce_warp<32, Op, reduce_size, thread_dim>(val);
    else
        return block_reduce_warp<64, Op, reduce_size, thread_dim>(val);
}
```
This **dynamically selects** based on runtime `warpSize`:
- **warpSize == 64** (GFX8/GFX9): template instantiated with WARP_SIZE=64
- **warpSize == 32** (GFX10+): template instantiated with WARP_SIZE=32

**Warp reduce** (`warp_reduce.hpp` line 51-56): Uses `__shfl_down`:
```cpp
for(auto d = warpSize / 2; d >= 1; d >>= 1)
    BinaryFunc<Op, FLOAT_ACCUM>{}.exec(val, __shfl_down(val, d));
```
On GFX9: 6 iterations (d = 32, 16, 8, 4, 2, 1). On RDNA: 5 iterations.

**Block reduce warp** (`block_reduce.hpp` lines 44-67):
```cpp
static __shared__ FLOAT_ACCUM shared[reduce_size / WARP_SIZE];
val = warp_reduce<Op>(val);        // intra-wavefront via __shfl_down
if(lane == 0) shared[wid] = val;   // one value per wavefront to LDS
__syncthreads();
val = tid < reduce_size / WARP_SIZE ? shared[lane] : 0;
if(wid == 0) val = warp_reduce<Op>(val);  // final reduction in first wavefront
```
For 256 threads on GFX9: `reduce_size/64 = 4` LDS entries = **16 bytes of LDS**.

### ReduceCalculation: Thread-to-Data Mapping
- Non-parallel path (line 93): `gid = threadIdx.x + blockIdx.x * blockDim.x`, one thread per output element.
- Input index: `(gid / inner_size) * inner_size * reduce_size + gid % inner_size` (line 97).
- Stride over reduce dimension: `input_idx += inner_size` (line 108) -- **strided access**, not contiguous in the reduce dimension.

### ReduceExtreme (argmin/argmax)
- Same 1-thread-per-output pattern.
- No wavefront reduction -- purely sequential loop per thread (line 52: `for(k = 1; k < reduce_size; ++k)`).
- No shared memory or inter-thread communication.

---

## Summary Table: Configuration Patterns

| Kernel | WG Size | Grid Strategy | Reduction Method | LDS Usage | Vector Loads | AMD-Specific |
|--------|---------|---------------|------------------|-----------|--------------|--------------|
| **LayerNorm** | 256 | 1 WG per row | LDS tree (8 steps) | 2048 B | None | None |
| **GroupNorm** | 1024 | 1 WG per group | LDS tree (10 steps) | 8192 B | None | None |
| **Softmax (OCL)** | 256 | CSR-Vector or CSR-Stream | LDS tree (8 steps) | 1024 B | None | None |
| **Softmax Attn (HIP)** | 64-256 dynamic | Warp/Block/Common select | DS_SWIZZLE + DS_BPERMUTE | 16 B (block) or 0 (warp) | None | Full: swizzle, bpermute, wavefront-size branching |
| **BatchNorm Train** | 256-1024 tuned | Per-channel (single) or 3D grid (multiple) | DPP asm (GCN) or LDS tree | 128 B (GCN) / 8192 B (non-GCN) | float2/4/8 | DPP row_shr, row_bcast, interleaved DPP |
| **BatchNorm Infer** | 2D tuned | Elementwise, no reduction | None | 0 | OpenCL vector types | None |
| **ReduceSum** | 256 | 1 or 2 kernels | __shfl_down + LDS (2-level) | 16 B | None | warpSize-branching (64 vs 32) |
| **ReduceCalc** | 256 | 1 thread/output | None (serial loop) | 0 | None | None |
| **ReduceExtreme** | 256 | 1 thread/output | None (serial loop) | 0 | None | None |

## Key Findings for IREE Configuration Design

1. **Workgroup sizes are mostly fixed constants** (256 or 1024), not dynamically computed from problem shape. The exception is the softmax attention solver which clamps to `[warpSize, 256]` based on seq_len.

2. **Wavefront-64 awareness is inconsistent**: The softmax attention kernel and batchnorm DPP paths are fully wavefront-aware. LayerNorm, GroupNorm, and the legacy softmax are completely wavefront-unaware -- they use the same LDS tree reduction regardless of wavefront width.

3. **The two-level reduction pattern** (wavefront intrinsics -> LDS for inter-wavefront) appears in the newest code (block_reduce.hpp, SoftmaxAttn) and is dramatically more efficient: 16 bytes of LDS vs 8192 bytes, and fewer barriers.

4. **AMD's key cross-lane instructions**:
   - `__shfl_down(val, offset)` -- works on both GFX9 (64-wide) and RDNA (32-wide)
   - `__hip_ds_swizzlef_N<op>` -- butterfly reduction within 32 lanes, single cycle
   - `__builtin_amdgcn_ds_bpermute(idx, val)` -- arbitrary permute across 64 lanes via LDS hardware
   - DPP `row_shr`, `row_bcast` -- GCN-only assembly, most efficient for GFX8/GFX9

5. **Vector loads are only used in BatchNorm** (float2/4/8), and only in the NHWC path. All other kernels use scalar loads. This is a missed optimization opportunity.

6. **Occupancy targeting** uses `GetMaxComputeUnits() * 4` as minimum number of workgroups (layernorm utils) or `GetMinWorkgroups()` at ~80% CU utilization (batchnorm). The softmax OCL path uses the magic number `64 * 40 * 8 = 20480` for grid capping.

7. **The coalescing unit on AMD GFX9 is 64 elements wide** (one element per lane in a wavefront). All the stride-loop patterns (`for(i = lid; i < size; i += LOCAL_SIZE)`) naturally achieve this since consecutive threads (lanes 0-63 within a wavefront) access consecutive memory addresses.