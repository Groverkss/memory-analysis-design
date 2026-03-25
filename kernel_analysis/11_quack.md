# Quack (Dao-AILab): Memory-Bound Kernel Analysis

---

## Exhaustive Memory-Bound Kernel Analysis: Quack (Dao-AILab)

**Important note**: Quack does **not** use Triton. All kernels are written with the NVIDIA CuTe DSL (via the `cutlass.cute` Python frontend). There are no `@triton.jit` decorators or `.cu`/`.cuh` files. The kernels compile to CUDA via CuTe's MLIR-based compilation pipeline.

---

### 1. RMSNorm Forward

**File**: `/home/kunwar/Work/kernel_libs/quack/quack/rmsnorm.py`, lines 26-287

**Block/Tile Sizes** (from `ReductionBase._get_tiled_copy`, line 28-36 of `reduction_base.py`):
- `num_threads`: 128 if N <= 16384, else 256 (line 23, `reduction_base.py`)
- `threads_per_row` (lines 33-38):
  ```
  N <= 64:    8 threads
  N <= 128:  16 threads
  N <= 3072: 32 threads
  N <= 6144: 64 threads
  N <= 16384: 128 threads
  N > 16384: 256 threads
  ```
- `tiler_mn = (num_threads // threads_per_row, vecsize * num_blocks_N * threads_per_row)` -- rows-per-CTA x cols-per-CTA
- `vecsize = 128 // largest_dtype_width` (128-bit loads, so 8 for fp16/bf16, 4 for fp32)
- Rows per CTA = `num_threads / threads_per_row`. E.g., for N=4096 bf16: 128/32 = 4 rows/CTA

**Cluster sizes** (lines 40-52):
- For 16-bit dtypes: `N <= 16K: 1, <= 32K: 2, <= 64K: 4, <= 128K: 8, else 16`
- For 32-bit dtypes: `N <= 32K: 1, <= 64K: 2, <= 128K: 4, <= 256K: 8, else 16`

**Program ID Mapping** (line 87):
- `grid = [ceil_div(M, tiler_mn[0]), cluster_n, 1]`
- `bidx` maps to rows, `cluster_y` to column chunks across CTA cluster

**Memory Access Patterns**:
- **cp.async** for global -> shared memory (128-bit vectorized, line 175)
- Shared memory layout: `order=(1, 0)` -- row-major in smem (column-contiguous in the N dimension for coalescing)
- Data reload from smem when register pressure is too high: `reload_from = "smem"` if `N > 8192` (RMSNorm) or `N > 16384` (LayerNorm) (line 30)
- Weight loaded from global directly to registers (not async), bias similarly

**Reduction Strategy**:
- **RMSNorm**: Single-pass `sum(x*x)` row reduction (line 243-252)
- **LayerNorm**: Two-pass -- first `sum(x)` for mean, then `sum((x-mean)^2)` for variance (lines 200-239)
- Row reduction is warp-level first (via `warp_reduction` with `threads_in_group=min(threads_per_row, 32)`), then block-level via shared memory buffer, then cluster-level via `mbarrier` + distributed shared memory stores

**Fusion**: RMSNorm fuses: residual add + norm computation + weight multiply + bias add + rstd/mean output into a single kernel. Full pattern: `out = (x + residual - mean) * rstd * weight + bias`

**Constants**:
- `eps` (default 1e-6)
- `vecsize = 128 // dtype_width` (8 for bf16, 4 for fp32)
- `smem_layout order = (1, 0)` (row-major)

---

### 2. RMSNorm Backward

**File**: `/home/kunwar/Work/kernel_libs/quack/quack/rmsnorm.py`, lines 494-774+

**Block/Tile Sizes**:
- `num_threads`: 128 if N <= 4096, else 256 (line 503-504)
- `threads_per_row` (lines 506-511): **Different** from forward:
  ```
  N <= 64:   8
  N <= 128: 16
  N <= 256: 32
  N <= 512: 64
  N <= 4096: 128
  N > 4096: 256
  ```
- 2 reduction stages for double-buffering

**Cluster sizes** (lines 513-519): Smaller thresholds than forward:
```
N <= 8K: 1, <= 16K: 2, <= 32K: 4, <= 64K: 8, else 16
```

**Program ID Mapping**:
- `grid = [sm_count, cluster_n, 1]` -- **persistent kernel** that loops over rows
- Each CTA processes `bidx_start, bidx_start + gdim, bidx_start + 2*gdim, ...` rows (line 688)

**Memory Access Patterns**:
- Double-buffered smem: `smem_layout = (tiler_mn[0], tiler_mn[1], 2)` with `order=(1,0,2)` (line 586)
- Prefetches next batch while computing current (cp.async pipelining, lines 690-709)
- `reload_wdy = "smem"` if `N > 16K` (line 498)

**Reduction**: `mean(x_hat * wdy)` row reduction via block/cluster reduce (lines 729-740)

**Fusion**: Fuses all of: x_hat computation, wdy = dout*w, mean reduction, dx = (wdy - x_hat * mean_xhat_wdy) * rstd, dW accumulation, dB accumulation, residual gradient add -- all in one persistent kernel

---

### 3. Softmax Forward

**File**: `/home/kunwar/Work/kernel_libs/quack/quack/softmax.py`, lines 24-165

**Block/Tile Sizes**:
- `num_threads`: 128 if N <= 16384, else 256
- `threads_per_row` (lines 36-39): Same table as RMSNorm forward
- `vecsize = 128 // largest_dtype_width`

**Cluster sizes** (lines 42-52): Same as RMSNorm forward

**Program ID Mapping**: `grid = [ceil_div(M, tiler_mn[0]), cluster_n, 1]`

**Memory Access**: cp.async global->smem (128-bit), smem->registers via `autovec_copy`, OOB filled with `-inf`

**Reduction Strategy** -- Two variants:
1. **Online softmax** (`online_softmax=True`, default): Single-pass `online_softmax_reduce` (file `reduce.py`, line 127-222) that computes max and sum_exp simultaneously using packed `f32x2_to_i64` encoding to combine max and sum into a single 64-bit value for cross-warp/cross-cluster communication
2. **Two-pass** (`online_softmax=False`): First pass for `max`, second pass for `sum(exp(x-max))`

**Key computation** (line 161): `y = exp_x * rcp_approx(denom)` -- uses approximate reciprocal for division

**Constants**: `log2_e = 1.4427` used for `exp2(x * log2e)` fast path

---

### 4. Softmax Backward

**File**: `/home/kunwar/Work/kernel_libs/quack/quack/softmax.py`, lines 221-349

**Block/Tile Sizes**: Same pattern, slight difference:
- `threads_per_row` (lines 227-229): boundary at 8192 instead of 16384 for 128 threads
- `num_threads`: 128 if N <= 8192, else 256 (line 246)

**Reduction**: Computes `dot = sum(dy * y)` via row_reduce, then `dx = y * (dy - dot)`

**Fusion**: Loads both dy and y via cp.async in parallel (2 async copies), computes everything in one pass

---

### 5. Cross-Entropy Forward

**File**: `/home/kunwar/Work/kernel_libs/quack/quack/cross_entropy.py`, lines 26-242

**Block/Tile Sizes**: Same as softmax forward. Same `threads_per_row` table (line 38-43).

**Cluster sizes**: Same as softmax (lines 45-55), except fp32 thresholds differ slightly: `N <= 16K: 1, <= 64K: 2, <= 128K: 4, <= 256K: 8`

**Program ID Mapping**: Same as softmax

**Reduction**: Same online_softmax or two-pass as softmax, but also computes:
- `lse = max_x + log(denom)`
- `loss = lse - target_logit`

**Fusion**: Fuses softmax reduction + loss computation + optional gradient computation (dx) into one kernel. When `has_dx=True`, also computes `probs = exp_x * rcp_approx(denom)` and `dx[target] = probs - 1`, `dx[other] = probs`.

**Special**: `reload_from = "smem"` if `N > 16384` and not online_softmax (line 36)

---

### 6. Cross-Entropy Backward

**File**: `/home/kunwar/Work/kernel_libs/quack/quack/cross_entropy.py`, lines 389-530

**Block/Tile Sizes**:
- Splits by blocks of 16K columns (line 396-404)
- `num_threads`: 128 if N <= 16384, else 256
- Tiler: `(cols_per_block, min(N, 16384))` -- tiles in the N dimension

**Program ID Mapping**: `grid = [ceil_div(M, tiler_mn[0]), ceil_div(N, tiler_mn[1]), 1]` -- 2D grid over both M and N

**No cluster** (no `_set_cluster_n` override, defaults to 1)

**Computation**: `probs = exp2(x * log2e - lse * log2e)`, then `grad = probs` or `probs - 1` at target index, scaled by `dloss`

---

### 7. RMS Final Reduce

**File**: `/home/kunwar/Work/kernel_libs/quack/quack/rms_final_reduce.py`, lines 26-118

A lightweight reduction kernel for the GEMM+RMS fused pipeline.

**Block/Tile Sizes**: Same as other kernels (same `threads_per_row` table). No cluster (`cluster_n = 1`).

**Program ID Mapping**: `grid = [ceil_div(M, tiler_mn[0]), 1, 1]` -- 1D over rows

**Computation**: `rstd = rsqrt(sum_n(x[m,n]) * scale + eps)` -- single row reduction of partial sums, then rsqrt

---

### 8. TopK Forward

**File**: `/home/kunwar/Work/kernel_libs/quack/quack/topk.py`, lines 25-216

**Constraints**: N must be power of 2, N <= 4096, k must be power of 2, k <= 128

**Block/Tile Sizes** (lines 37-42):
- `threads_per_row = max(min(N // k, 32, N // 64), 1)` -- ensures each thread holds >= k elements
- `num_threads`: 128 (N <= 16384)
- `vecsize = 128 // dtype_width`

**Program ID Mapping**: `grid = [ceil_div(M, tiler_mn[0]), 1, 1]`

**Algorithm**: Bitonic sort-based top-k:
1. Encode column indices into bottom bits of float32 values (lines 112-127)
2. Call `bitonic_topk` from `quack/sort/bitonic_sort.py` with `warp_width=threads_per_row`
3. Optional softmax on top-k values (lines 169-186): inline max + exp2 + warp_reduction_sum + rcp_approx

**Memory Access**: Standard 128-bit vectorized load, direct register-to-global vectorized write for output

---

### 9. TopK Backward

**File**: `/home/kunwar/Work/kernel_libs/quack/quack/topk.py`, lines 292-457

**Block/Tile Sizes**: Same pattern. `num_threads`: 128 if N <= 16384, else 256.

**Strategy**: Scatter-based -- writes gradients into shared memory at target indices, then copies smem to global:
1. Zero smem
2. Load dvalues, values, indices
3. If softmax: compute `dot = sum(dvals * vals)`, then `grads = vals * (dvals - dot)` (same as softmax backward)
4. Scatter `grads[i]` to `sdX[row, index[i]]` in shared memory (line 448)
5. Read back smem -> registers -> global

---

### 10. Activation Functions (Epilogue Components)

**File**: `/home/kunwar/Work/kernel_libs/quack/quack/activation.py`

These are **not standalone kernels** but are fused into GEMM epilogues. Key implementations:

- **SiLU** (line 240): `silu(x) = 0.5*x*tanh(0.5*x) + 0.5*x` -- 3 SASS instructions (FMUL, MUFU.TANH, FFMA)
- **SwiGLU** (line 256): `silu(x) * y` -- fused gate+up projection activation
- **dSwiGLU** (line 263): Optimized backward: 1 MUFU.TANH, 5 FMUL, 3 FFMA
- **GELU tanh approx** (line 113): `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))`
- All have packed `f32x2` variants for SM100+ that process 2 elements at once

---

### 11. GEMM + Norm + Activation (Fused)

**File**: `/home/kunwar/Work/kernel_libs/quack/quack/gemm_norm_act.py`

Fuses GEMM output normalization and activation into the GEMM epilogue:
- `PostAct = act((A @ B + C) * colvec * rowvec)` where colvec = rstd, rowvec = norm_weight

### 12. GEMM + Squared Reduction

**File**: `/home/kunwar/Work/kernel_libs/quack/quack/gemm_sq_reduce.py`

Fuses column-vector reduction of squared GEMM output into the epilogue:
- `reduce[m] = sum_n(D_raw[m,n]^2)`, then `D_out = D_raw * rowvec`
- Squared sum computed **before** rowvec scaling

---

### 13. Stochastic Rounding (Quantization Primitive)

**File**: `/home/kunwar/Work/kernel_libs/quack/quack/rounding.py`

- FP32 -> BF16 stochastic rounding using Blackwell `cvt.rs.satfinite.bf16x2.f32` PTX instruction
- Uses Philox 4x32b PRNG (7 rounds default) for random bits
- Processes elements in pairs; 4 pairs share one Philox call (4 random words)

---

### Common Patterns Across All Kernels

**Vectorization**: All kernels use 128-bit loads (`vecsize = 128 // dtype_width`): 8 elements for bf16/fp16, 4 for fp32.

**Shared memory layout**: Always `order=(1, 0)` (N-contiguous in smem, matching global memory's row-major layout for coalesced loads).

**Copy infrastructure** (`copy_utils.py`):
- `tiled_copy_2d(dtype, threads_per_row, num_threads, vecsize)` creates a 2D thread-to-data mapping with layout `(rows, cols)` where `rows = num_threads // threads_per_row`
- Predicates handle non-even N via `predicate_k`

**Reduction hierarchy** (all in `reduce.py`):
1. Thread-local: `TensorSSA.reduce()` across per-thread elements
2. Warp-level: `warp_reduction(val, op, threads_in_group=min(threads_per_row, 32))`
3. Block-level: Via shared memory buffer `reduction_buffer[num_warps/warps_per_row, warps_per_row]`
4. Cluster-level: Via `mbarrier` + `store_shared_remote` (distributed shared memory)

**The `threads_per_row` table** is the single most important tuning parameter -- it appears identically in RMSNorm fwd, Softmax fwd/bwd, Cross-Entropy fwd, and RMS Final Reduce:
```
N <= 64:    8
N <= 128:  16
N <= 3072: 32
N <= 6144: 64
N <= 16384: 128
N > 16384: 256
```