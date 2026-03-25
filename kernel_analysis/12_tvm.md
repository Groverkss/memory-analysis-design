# Apache TVM: Memory-Bound Codegen Analysis

## Source: https://github.com/apache/tvm.git

---

## TVM GPU Code Generation for Memory-Bound Ops: Exhaustive Analysis

### Architecture Overview

TVM's GPU code generation for memory-bound ops operates through two main systems:

1. **DLight (Default Light) scheduling rules** -- the primary system used in production pipelines. These are deterministic, hand-tuned schedule templates applied via `ApplyDefaultSchedule`.
2. **MetaSchedule** -- a tuning-based system that explores a search space of schedules. Uses `SampleCategorical` for non-deterministic exploration.

The pipeline ordering for CUDA/ROCm (from `/home/kunwar/Work/kernel_libs/tvm/python/tvm/relax/backend/cuda/pipeline.py`, line 41-47):
```python
dl.ApplyDefaultSchedule(
    dl.gpu.Matmul(),
    dl.gpu.GEMV(),
    dl.gpu.Reduction(),
    dl.gpu.GeneralReduction(),
    dl.gpu.Fallback(),
)
```

Rules are tried **in order**, first match wins (from `/home/kunwar/Work/kernel_libs/tvm/python/tvm/s_tir/dlight/base/transform.py`, line 87-93).

---

### 1. INNER REDUCTION (e.g., `sum(x, axis=-1)`, row-wise reduction)

**File:** `/home/kunwar/Work/kernel_libs/tvm/python/tvm/s_tir/dlight/gpu/reduction.py`, `_sch_inner_reduction` (lines 177-233)

**Strategy:**
- Spatial loops (S) are fused into one loop, reduction loops (R) are fused into one loop.
- The reduction loop is split: `[outer_r, threadIdx.x]`.
- `rfactor` on `threadIdx.x` creates a per-thread partial result.
- Per-thread results are in **local** scope.
- Final reduction is done via cross-thread reduction (lowered later to warp shuffles + shared memory).

**Thread binding:**
- **threadIdx.x** = reduction threads (split from the reduction loop)
- **blockIdx.x** = spatial (each row = one block)
- Thread count decided by `suggest_threads_per_block()`.

**Thread count logic** (from `/home/kunwar/Work/kernel_libs/tvm/python/tvm/s_tir/dlight/base/utils.py`, lines 66-112):
- Start with budget: **CUDA=1024, ROCm=256, Metal=256, OpenCL=256, else 64**
- For each loop, find largest power of 2 that fits both the loop extent and remaining thread budget.
- Dynamic loops get `max_threads_for_dynamic_loop=32` as base, then remaining budget goes to the first dynamic loop.

**Epilogue handling:**
- If epilogue is a broadcast (e.g., normalization: reduce then broadcast result back), the reduction result is placed in **shared memory**, and the epilogue is parallelized across `threadIdx.x`.
- If element-wise epilogue, result stays in **local**.

**Vectorization:** None explicit in this rule.

**Unrolling:** `pragma_auto_unroll_max_step=256`, `pragma_unroll_explicit=1`.

---

### 2. OUTER REDUCTION (reduction on non-innermost axis, e.g., `sum(x, axis=0)`)

**File:** `/home/kunwar/Work/kernel_libs/tvm/python/tvm/s_tir/dlight/gpu/reduction.py`, `_sch_inner_spatial` (lines 235-303)

**Strategy:**
- Fixed tile sizes: **len_tx=16, len_ty=16** (hardcoded, line 248).
- Spatial dimension split by `len_tx`, reduction dimension split by `len_ty`.
- `rfactor` on `threadIdx.y`.
- Per-thread partial results in **local**.

**Thread binding:**
- **threadIdx.x** = spatial (inner dimension, up to 16, adjusted down to be a factor of the spatial extent -- lines 252-255)
- **threadIdx.y** = reduction (16 threads)
- **blockIdx.x** = spatial outer
- Total threads per block = up to 16*16 = 256.

**Epilogue handling:**
- Broadcast epilogue: reduction result to **shared**, epilogue parallelized with `[threadIdx.x, threadIdx.y]`.
- Element-wise: epilogue bound to `threadIdx.x` only.

---

### 3. GENERAL REDUCTION (softmax, layer_norm, RMS norm -- multi-block patterns)

**File:** `/home/kunwar/Work/kernel_libs/tvm/python/tvm/s_tir/dlight/gpu/general_reduction.py` (entire file)

**This is the most important rule for your use case.** It handles ops that decompose into multiple blocks -- like softmax (max, exp, sum, normalize) and norms (reduce-sum, reduce-sum-sq, rsqrt, normalize).

**Strategy:**
- All blocks are identified. The pattern is `SSSR...` -- leading spatial dims, trailing reduction dims.
- All blocks in the chain are computed at the `blockIdx.x` level.
- Each block's reduction dimension is split: `[outer_r, threadIdx.x]` where `threadIdx.x = len_tx`.
- Intermediate results between blocks (reductions) go to **shared memory**.

**Thread count constants (line 40-48):**
```python
if target.kind.name == "cuda":
    len_tx = 256
    unroll_depth = 256
elif target.kind.name == "opencl":
    len_tx = 256
    unroll_depth = 64
else:
    len_tx = 64
    unroll_depth = 64
```

**Thread binding (lines 159-177):**
- Leading spatial loops fused into `blockIdx.x`.
- For each block (iterated in reverse order):
  - Trailing reduction loops are fused.
  - Split into `[outer_r, len_tx]`.
  - Inner part bound to `threadIdx.x`.
  - Outer part gets unroll annotation.
- Intermediate buffers set to **shared** scope.

**Key insight:** One block per "row" (spatial element). **All 256 threads cooperate on the reduction dimension within each block.** Inter-block sync via shared memory writes + `__syncthreads` (implicit from the shared scope).

**No vectorization** in this rule.

**Unrolling:** `pragma_auto_unroll_max_step=256` (CUDA) or 64 (others).

---

### 4. RMS NORM (specialized rule)

**File:** `/home/kunwar/Work/kernel_libs/tvm/python/tvm/s_tir/dlight/gpu/rmsnorm.py`

**This is a hand-tuned schedule specifically for RMS normalization.**

**Thread count (lines 83-88):**
```python
if target.kind.name == "cuda":
    num_tx = 512
elif target.kind.name == "opencl":
    num_tx = 256
else:
    num_tx = 64
```

**Strategy:**
1. Cache input to **local** memory.
2. Inline cast/load blocks.
3. Remaining blocks: read, sqr, redsum, rsqrt, norm, write.
4. Each is fused across batch dims.
5. Read and norm are split: `[num_tx, outer, 8]` -- **vectorize width = 8**.
6. rsqrt output goes to **shared** memory (broadcast pattern).
7. sqr and redsum stay in **local** scope.

**Vectorization:** Explicit **vector width = 8** for both read and write (lines 117-122, 129-131).

**Thread binding:**
- `blockIdx.x` = batch/row dimension
- `threadIdx.x` = 512 threads on CUDA (split from inner dimension)

**Shared memory:** Only the rsqrt result (one value per row) is in shared memory. This is the broadcast value.

---

### 5. GEMV INNER REDUCTION (vector-matrix multiply, also applies to single-row reductions)

**File:** `/home/kunwar/Work/kernel_libs/tvm/python/tvm/s_tir/dlight/gpu/gemv.py`, `sch_inner_reduction` (lines 87-425)

**Target-specific constants for CUDA (lines 319-329):**
```python
VEC_C = 4           # vectorize factor for inner compute
LOAD_V_SHARED = True
LOAD_V_VEC = 8      # vector width for shared memory load
VEC_LOAD = 4        # vector width for weight load
UNROLL = 256
SUPPORT_WARP_SHUFFLE = True
TS, TR = 16, 32     # spatial threads, reduction threads (static shapes)
# or
TS, TR = 1, 64      # dynamic shapes
```

**For ROCm (lines 345-358):**
```python
VEC_C = 4
LOAD_V_SHARED = False   # Disabled due to ROCm issues
LOAD_V_VEC = 8
UNROLL = 256
TS, TR = 8, 64   # or 1, 128 if S > R
```

**Strategy:**
- Double rfactor: first reduce to `TS * TR * VEC_C`, then to `TS * TR`, then to `TS`.
- Vector input loaded to **shared memory** (CUDA, up to `max_shared_memory_per_block`).
- Weight loaded to **local** memory with vectorization.

**Thread binding:**
- `threadIdx.y` = spatial (TS)
- `threadIdx.x` = reduction (TR)
- `blockIdx.x` = spatial outer

**Vectorization:**
- Compute: vectorize `VEC_C=4` (CUDA)
- Weight load: `VEC_LOAD=4` (CUDA)
- Shared memory load: `LOAD_V_VEC=8` (CUDA)

---

### 6. FALLBACK (elementwise, pointwise, any unmatched op)

**File:** `/home/kunwar/Work/kernel_libs/tvm/python/tvm/s_tir/dlight/gpu/fallback.py`

**Strategy:**
- Inline all possible blocks.
- For remaining blocks: fuse all spatial loops, split into `[blockIdx.x, threadIdx.x]`.
- `threadIdx.x = max_threads_per_block` (1024 for CUDA, 256 for others).
- Any reduction loops: `decompose_reduction` (serial within each thread).

**This is the schedule for elementwise ops.** One thread per element (up to `max_threads_per_block` per block).

**No vectorization, no shared memory, no tiling.**

---

### 7. TRANSPOSE

**File:** `/home/kunwar/Work/kernel_libs/tvm/python/tvm/s_tir/dlight/gpu/transpose.py`

**Constants for CUDA (lines 52-56):**
```python
len_tx = 16
len_ty = 8
unroll_depth = 256
len_vec = 4
```

**Strategy:**
- Uses **shared memory** with `storage_align(factor=32, offset=1)` to avoid bank conflicts.
- 2D thread tiling: `blockIdx.y * blockIdx.x * threadIdx.y(8) * threadIdx.x(16)`.
- Shared memory tile = 16 x 8 x c_factor.
- Vectorization up to `len_vec=4`.

---

### 8. HOW TVM LOWERS CROSS-THREAD REDUCTIONS TO GPU PRIMITIVES

**File:** `/home/kunwar/Work/kernel_libs/tvm/src/s_tir/transform/lower_thread_allreduce.cc`

Two strategies based on `IsWarpReduction`:

**a) Warp-level reduction (preferred when reduce_extent <= max_threads and is contiguous):**
- If `reduce_extent <= warp_size` (32): Single warp shuffle-down reduction + broadcast via `tvm_warp_shuffle`.
- If `reduce_extent > warp_size` and is a multiple of warp_size: Two-stage reduction:
  1. **Intra-warp shuffle-down** (each warp reduces its portion)
  2. Write partial results to **shared memory** (one per warp)
  3. Sync threads
  4. **Second shuffle-down** across warp leaders
  5. Broadcast result via shared memory

**b) Shared-memory tree reduction (fallback):**
- All values written to shared memory buffer of size `group_extent * reduce_extent`.
- Tree-based parallel reduction with `__syncthreads()` between levels.

---

### 9. OP DEFINITIONS (how ops decompose)

**Softmax** (`/home/kunwar/Work/kernel_libs/tvm/include/tvm/topi/nn/softmax.h`, lines 50-116):
- 4 stages: `max(input)` -> `exp(input - max)` -> `sum(exp)` -> `exp / sum`
- 2 reductions (max and sum), 2 elementwise (exp, divide)
- Matched by `GeneralReduction` rule

**Layer Norm** (`/home/kunwar/Work/kernel_libs/tvm/include/tvm/topi/nn/layer_norm.h`, lines 51-133):
- Uses `MakeTupleSumReducer` to compute `sum(x)` and `sum(x^2)` simultaneously in one reduction pass
- Then: `mean = sum_x / N`, `var = sum_x2 / N - mean^2`, `output = (x - mean) * rsqrt(var + eps) * gamma + beta`
- fp16 inputs cast to fp32 for reduction
- Tagged `kInjective` (not `kCommReduce`) for the final output

**RMS Norm** (`/home/kunwar/Work/kernel_libs/tvm/include/tvm/topi/nn/rms_norm.h`, lines 50-104):
- `square = x * x` -> `sum(square)` -> `rsqrt(sum/N + eps)` -> `rsqrt * x * weight`
- Always casts to fp32 for computation, casts back to input dtype at the end.

---

### 10. MetaSchedule AUTO-TUNING APPROACH

MetaSchedule uses these schedule rules for search:

**CrossThreadReduction** (`cross_thread_reduction.cc`):
- Thread extent candidates sampled via `SampleCategorical` (uniform distribution across candidates).
- Default candidates: `[32, 64, 128, 256, 512, 1024]` (from `auto_bind.py` defaults).
- If fusible with consumer, computes-at the reduction to the consumer's spatial loop, sets output to **shared** scope.
- Splits the fused reduction loop: `[outer, thread_extent]`, binds inner to `threadIdx.x`.

**AutoBind** (`auto_bind.cc`):
- For spatial blocks: splits spatial loop into `[blockIdx.x, threadIdx.x]`.
- Thread extent sampled from candidates.
- Default `max_threadblocks=256`.

**ParallelizeVectorizeUnroll** (`parallel_vectorize_unroll.py`):
- `max_vectorize_extent=16` (default)
- `unroll_max_steps` defaults to empty (no unroll sampling)

---

### Summary Table: Thread Configurations

| Rule | blockIdx | threadIdx.x | threadIdx.y | Total threads/block | Vectorization |
|------|----------|-------------|-------------|---------------------|---------------|
| **GeneralReduction** (softmax, norms) | fused spatial | 256 (CUDA), 256 (OpenCL), 64 (other) | -- | 256 | None |
| **RMSNorm** (specialized) | batch/row | 512 (CUDA), 256 (OpenCL) | -- | 512 | **8** (read + write) |
| **Reduction inner** | fused spatial | from `suggest_threads_per_block` (up to 1024) | -- | up to 1024 | None |
| **Reduction outer** | fused spatial | 16 (spatial) | 16 (reduction) | 256 | None |
| **GEMV inner (CUDA)** | spatial outer | 32 (reduction) | 16 (spatial) | 512 | VEC_C=4, VEC_LOAD=4, LOAD_V_VEC=8 |
| **GEMV inner (ROCm)** | spatial outer | 64 (reduction) | 8 (spatial) | 512 | VEC_C=4 |
| **Fallback** (elementwise) | spatial outer | 1024 (CUDA), 256 (other) | -- | 1024 | None |
| **Transpose (CUDA)** | 2D grid | 16 | 8 | 128 | 4 |

### Key Patterns for IREE

1. **No shape-dependent configuration**: TVM uses fixed constants per target (e.g., 256 threads for GeneralReduction on CUDA). The only shape-dependent logic is in `suggest_threads_per_block`, which does power-of-2 fitting.

2. **Reduction decomposition is the key scheduling decision**: inner reduction uses rfactor + cross-thread allreduce; outer reduction uses 2D thread tiling (tx for spatial, ty for reduction).

3. **Shared memory usage is minimal for memory-bound ops**: mainly for inter-block communication in multi-stage ops (softmax, norms) and for cross-thread allreduce staging. Not for explicit data reuse.

4. **Vectorization is only in specialized rules**: RMSNorm uses vector width 8; GEMV uses 4-8. GeneralReduction and basic Reduction have **no vectorization at all**. This seems like a performance gap.

5. **The GeneralReduction rule is a catch-all for fused norm/softmax patterns**: it works by putting each reduction stage in shared memory and parallelizing each reduction loop with `len_tx` threads. Simple but effective.

6. **MetaSchedule search space is narrow for memory-bound ops**: just thread extent sampling from a few candidates. The real intelligence is in the DLight rules.