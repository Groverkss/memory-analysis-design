# Cross-Cutting Lessons for Memory-Bound Kernel Configuration

## Synthesized from analysis of 12 kernel libraries/compilers

This document extracts the key configuration patterns that appear consistently across
NVIDIA Apex, PyTorch, llama.cpp, MIOpen, Liger-Kernel, Unsloth, FlashInfer, Quack,
ThunderKittens, CUTLASS, CUB/CCCL, and TVM. These patterns should inform the v2
memory-bound configuration system for IREE.

---

## 1. THE UNIVERSAL THREAD BLOCK SIZES

Every library converges on a small set of block sizes:

| Block Size | Used For | Libraries |
|-----------|----------|-----------|
| **128** | Norm forward, softmax (persistent warp), elementwise | Apex, PyTorch, FlashInfer (CuTe), Quack |
| **256** | General reductions, norm, elementwise, softmax | PyTorch, llama.cpp, MIOpen, CUB, TVM, CUTLASS |
| **512** | Large reductions, batch norm, RMSNorm (TVM) | Apex (BN/GN), PyTorch (softmax/reduce), CUB (SM100), TVM |
| **1024** | Very large reductions, elementwise (TVM fallback) | llama.cpp (large ncols), PyTorch (spatial softmax), MIOpen (GroupNorm) |

**Lesson:** Block sizes are NOT tuned per-problem. They're categorical choices:
- **128** for norm/softmax where reduction fits in a few warps
- **256** as the safe default for everything
- **512-1024** only when the reduction dimension is very large (>4K elements)

**For IREE:** Don't overthink block size selection. Start with 128 for norms, 256 for general reductions, and have a simple escalation rule based on reduction dimension size.

---

## 2. THE UNIVERSAL VECTOR LOAD WIDTH: 128 BITS

**Every single library** uses 128-bit (16-byte) loads as the maximum single-instruction load:
- Apex: `BYTES_PER_LDG=16`, `uint4` (128 bits)
- PyTorch: `aligned_vector<T, vec_size>` where vec_size gives 128-bit loads
- FlashInfer: `vec_t<T, VEC_SIZE>` backed by `uint4`, `vec_size = 16 / sizeof(T)`
- CUB: `VECTOR_LOAD_LENGTH=4` (4 x float = 128 bits)
- CUTLASS: `AlignedArray` with natural alignment, 128-bit preferred
- Quack: `vecsize = 128 // dtype_width` (8 for bf16, 4 for fp32)
- ThunderKittens: `float4` (16 bytes) per thread for global loads
- TVM: Vector width 8 for RMSNorm (8 x fp16 = 128 bits)
- llama.cpp: `float2` (8 bytes for f32), no 128-bit -- **the exception that proves the rule** (llama.cpp is not fully optimized for vectorized access)

**Elements per 128-bit load by type:**
- fp16/bf16: 8 elements
- fp32: 4 elements
- fp64: 2 elements
- int8: 16 elements

**Lesson:** The coalescing target for v2 should always be based on 128-bit loads. The `vectorWidth` in IREE's config should be `128 / elementBitWidth`.

**For IREE:** v1's `maxLoadBits = 128` is correct. The coalescing target formula `subgroupSize * (maxLoadBits / elemBits)` is the right approach.

---

## 3. ELEMENTS PER THREAD: THE KEY SCALING PARAMETER

CUB's `MemBoundScaling` formula is the gold standard for adapting work-per-thread to data type:

```
items_per_thread = clamp(nominal_4B * 4 / sizeof(T), 1, 2 * nominal_4B)
```

This keeps **total bytes per tile roughly constant** (~16KB for types >= 4 bytes):

| sizeof(T) | Items/Thread (nominal=16) | Tile Size (bytes) |
|-----------|--------------------------|-------------------|
| 1 | 32 | 8192 |
| 2 | 32 | 16384 |
| 4 | 16 | 16384 |
| 8 | 8 | 16384 |
| 16 | 4 | 16384 |

**Across libraries:**
- Apex LayerNorm: `n2/128` elements/thread = hidden_dim / block_size
- PyTorch elementwise: 8 elements/thread (float), 16 (byte)
- CUB reduce: 16 items/thread (4B), scaled by type
- Liger-Kernel: entire row in one shot (BLOCK_SIZE = hidden_dim)
- FlashInfer RMSNorm: `ceil(d/threads) * vec_size` per thread

**Lesson:** The number of elements per thread should scale inversely with element size to keep bytes-per-thread roughly constant. A good default is ~64-128 bytes per thread.

**For IREE:** The budget check should work in bytes, not elements. `perThreadBytes = vectorWidth * elementBytes * numIterations` should target ~64-128 bytes.

---

## 4. REDUCTION DECOMPOSITION: THE THREE-LEVEL PATTERN

Every library uses the same three-level reduction:

### Level 1: Thread-Local Accumulation
Each thread accumulates across its assigned elements sequentially. Multiple independent accumulators (ILP=4 in PyTorch, PARALLEL_LOADS=4 in Apex, vt0=4 in CUB) hide memory latency.

### Level 2: Warp-Level Shuffle
- NVIDIA: `__shfl_xor_sync` or `__shfl_down_sync`, 5 steps for 32-thread warp
- AMD GFX9: DPP `row_shr` + `row_bcast` (6 steps for 64-thread wavefront), or `__shfl_down` (6 steps)
- AMD RDNA: `__shfl_down` (5 steps for 32-thread wavefront)
- **Zero shared memory needed** for this level

### Level 3: Cross-Warp via Shared Memory
- Each warp's lane 0 writes result to `shared[warp_id]`
- `__syncthreads()`
- First warp (or thread 0) reduces the `num_warps` partial results
- Broadcast result via `shared[0]`
- **Shared memory**: `num_warps * sizeof(accumulator)` (negligible: 16-64 bytes)

**AMD-specific optimization** (from MIOpen SoftmaxAttn):
- Level 2 uses `__hip_ds_swizzlef_N` (DS_SWIZZLE) for within-32-lane butterfly
- Uses `__builtin_amdgcn_ds_bpermute` for crossing 32-lane boundary in 64-wide wavefronts
- This achieves single-cycle cross-lane operations without LDS

**For IREE:** The reduction strategy is already well-understood. The key insight is that **shared memory usage for reductions is trivially small** (just `num_warps * sizeof(accum)`). The budget check shouldn't penalize reductions much. The real shared memory cost is for data caching (if needed).

---

## 5. ONE ROW PER BLOCK: THE DOMINANT PATTERN FOR NORMS/SOFTMAX

The overwhelming pattern across all libraries for normalization and softmax:

```
grid.x = num_rows  (batch * seq_len, or batch * spatial)
block.x = 128-512  (threads cooperate on the reduction within one row)
```

**When this breaks:**
- **Hidden dim too small** (< 128 elements): Multiple rows per block. Apex contrib: `WARPS_M=4` for hidden <= 2304. Liger-Kernel: `BLOCK_ROW=16` for BLOCK_SIZE <= 256. FlashInfer QKRMSNorm: 4 warps handle 4 heads independently within one block.
- **Hidden dim too large** (> ~8192 elements): Multiple blocks per row. Apex contrib: `CTAS_PER_ROW=2,4,8` with cooperative launch + global memory barriers. Quack: CTA clusters with distributed shared memory.

**The transition points:**

| Hidden Dim | Strategy | Block Config |
|-----------|----------|--------------|
| < 128 | Multiple rows/block (sub-warp or multi-warp) | Small block, many rows |
| 128 - 8192 | **One row per block** (sweet spot) | 128-512 threads |
| 8192 - 65536 | One row per block, large block or chunked reduction | 256-1024 threads, loop |
| > 65536 | Multiple blocks per row (cooperative) | Global sync needed |

**For IREE:** The workgroup tiling for the parallel dimension should usually produce `numWGs = product(parallel_dims)`. The threads-per-workgroup handles the reduction. Only deviate for very small or very large reduction dims.

---

## 6. COALESCING: THREADS MAP TO CONTIGUOUS MEMORY

Every single library achieves coalescing the same way:
- Thread `t` accesses elements at `base + t * vec_size` (contiguous)
- Stride between iterations: `num_threads * vec_size`

This means:
- **For innermost reduction** (e.g., sum along last dim): `threadIdx.x` maps to the reduction dimension. 32 consecutive threads read 32 * vec_size contiguous elements. Coalesced.
- **For outer reduction** (e.g., sum along first dim): `threadIdx.x` maps to the output (spatial) dimension. Reads from the reduction dim are strided. Coalescing comes from multiple threads reading adjacent output elements.

**The key AMD difference:** On GFX9, the coalescing unit is 64 lanes wide (full wavefront), not 32. So the minimum coalesced access is 64 * vec_size elements.

**For IREE:** v1's coalescing walk from innermost dimension outward is correct. The target should be `subgroupSize * vectorWidth` contiguous elements. On AMD GFX9, subgroupSize=64 gives target=512 for fp16 (64*8), vs NVIDIA's 256 (32*8).

---

## 7. THE TWO-PASS vs ONE-PASS TRADEOFF

For norms (mean + variance) and softmax (max + sum), libraries split between:

### Two-Pass (read data twice from global memory)
- Pass 1: Reduction (sum, max)
- Pass 2: Elementwise with reduction result (normalize, exp/divide)
- **Used by:** llama.cpp (all norms, softmax), MIOpen (all norms, softmax), PyTorch (fallback norm, large softmax), FlashInfer RMSNorm (CUDA version)

### One-Pass (cache in registers or shared memory)
- Load data once, compute reduction AND output
- Data stays in registers (Welford online for norms) or shared memory
- **Used by:** Apex (Welford), PyTorch (vectorized norm), Liger-Kernel (entire row in registers via Triton), FlashInfer (FusedAddRMSNorm caches in smem)

### Which is better?
- **Small hidden dim** (fits in registers): One-pass always wins
- **Large hidden dim** (doesn't fit): Two-pass is simpler and often similar speed (memory bandwidth dominated)
- Liger-Kernel/Unsloth: Always one-pass because `BLOCK_SIZE = hidden_dim` (entire row in registers). Limited to 65536 elements.
- Apex GroupNorm v2: Explicit `LOAD_TWICE` mode when data doesn't fit in registers -- accept 2x bandwidth for lower register pressure.

**For IREE:** The v1 approach of tiling the reduction dimension (`partial_reduction` tiles) and iterating is correct for large reductions. For small reductions that fit in registers, the per-thread work should be enough to hold everything. The key decision is whether to cache in shared memory between reduction and elementwise passes.

---

## 8. SHARED MEMORY USAGE IN MEMORY-BOUND OPS

Shared memory usage in memory-bound ops is **minimal** compared to compute-bound ops:

| Purpose | Typical Size | Who Uses It |
|---------|-------------|-------------|
| Cross-warp reduction buffer | 16-128 bytes | Everyone |
| Broadcast reduction result | 4-8 bytes per row | TVM, PyTorch norm |
| Full row cache (avoid re-read) | `hidden_dim * sizeof(T)` | FlashInfer FusedAddRMSNorm, PyTorch softmax smem path |
| Transpose tile | ~2KB (32x32 * sizeof(T)) | CUTLASS, TVM transpose |

**The key observation:** Most memory-bound kernels use < 256 bytes of shared memory. The exceptions are:
1. **Softmax with cached data** (PyTorch `cunn_SoftMaxForwardSmem`): Caches entire row in smem to avoid 3 global reads. `dim_size * sizeof(T)` bytes.
2. **FlashInfer FusedAddRMSNorm**: Caches intermediate `x` in smem. `d * sizeof(float)` bytes.
3. **Transpose**: Needs shared memory for reordering. `tile_size * sizeof(T)` bytes with +1 padding for bank conflicts.

**For IREE:** Shared memory budget is not a binding constraint for most memory-bound ops. The budget check in v1 should focus on register pressure (elements per thread), not shared memory. Reserve shared memory analysis for transpose and shared-memory-promotion cases.

---

## 9. THE `threads_per_row` TABLE (from Quack/FlashInfer)

Quack (Dao-AILab) and FlashInfer's CuTe-DSL RMSNorm share a remarkably similar heuristic for mapping threads to the reduction dimension:

```
N <= 64:    threads_per_row = 8
N <= 128:   threads_per_row = 16
N <= 3072:  threads_per_row = 32
N <= 6144:  threads_per_row = 64
N <= 16384: threads_per_row = 128
N > 16384:  threads_per_row = 256
```

With `total_threads = 128` (or 256 for large N), this gives:
- `rows_per_block = total_threads / threads_per_row`
- Small N -> many rows per block (better occupancy)
- Large N -> one row per block (all threads on one row)

**This is the most refined heuristic across all libraries for choosing the thread decomposition.**

**For IREE:** This table should inform the lane_basis / subgroup_basis production. When N is small, pack multiple rows into a workgroup. When N is large, dedicate all threads to one row. The transition points (64, 128, 3072, 6144, 16384) are battle-tested values.

---

## 10. BACKWARD KERNELS: PERSISTENT THREAD PATTERN

Every Triton library (Liger-Kernel, Unsloth, FlashInfer) uses the same backward kernel pattern:

```python
grid = (sm_count,)  # one program per SM
for row_idx in range(my_start, my_end):
    # process row, accumulate dW, dB
```

This "persistent thread" pattern avoids atomics for weight/bias gradient accumulation. Each SM exclusively owns a subset of rows and accumulates into thread-local registers.

**CUDA libraries** (PyTorch, Apex) use a different approach: partial reductions across blocks with global memory staging + atomics or a second kernel pass.

**For IREE:** This is relevant for fused backward dispatch configs. The grid should be `min(sm_count, num_rows)` to balance occupancy against atomic contention.

---

## 11. HOW V1 PROBLEMS MAP TO LIBRARY SOLUTIONS

### Problem: Elementwise transpose drops all coalescing (v1 Example 10)

**Library solutions:**
- PyTorch: Separate `cunn_SoftMaxForward` (coalesce input) vs spatial softmax (different thread mapping)
- TVM: Explicit `Transpose` rule with shared memory and bank-conflict-avoiding padding
- CUTLASS: `OutputTileOptimalThreadMap` with 256-byte access width target
- ThunderKittens: Shared memory tiles with swizzle patterns for bank conflict avoidance

**Lesson:** Conflicting coalescing requirements need shared memory promotion with bank-conflict-avoiding swizzle. This is a well-solved problem. IREE should detect coalescing conflicts and insert shared memory promotion (like `setTransposeConfig` already does).

### Problem: Budget doesn't account for thread-level parallelism in elementwise ops

**Library solutions:**
- PyTorch: `block_work_size = num_threads * thread_work_size` -- per-thread work is independent, no "budget" concept
- CUB: `MemBoundScaling` works in bytes-per-tile, not sequential work
- Liger-Kernel DyT: 2D grid tiles both dimensions independently for elementwise

**Lesson:** For elementwise ops, the "budget" is just bytes-per-thread, not total per-WG work. All tiles are independently parallel. There's no sequential work to bound.

### Problem: No lane_basis / subgroup_basis production

**Library solutions:**
- Quack's `threads_per_row` table maps directly to what lane_basis should be
- CUB's `set_block_dimension` logic (block_width from dim0, block_height from dim1)
- PyTorch Reduce.cuh `setReduceConfig`: block_width for reduce dim, block_height for output dim

**Lesson:** The lane_basis should distribute threads from the innermost contiguous dimension outward, exactly as v1 designed. The missing piece is just the implementation.

---

## 12. CONFIGURATION DECISION TREE (synthesized from all libraries)

```
Input: op type, tensor shapes, element types, target GPU

1. Classify op:
   a) Pure elementwise (no reduction) -> flat 1D grid, 128-256 threads, vec_size=128/elemBits
   b) Inner reduction (reduce along contiguous dim) -> one block per row
   c) Outer reduction (reduce along non-contiguous dim) -> 2D thread block
   d) Multi-stage (norm, softmax) -> same as inner reduction with shared memory for intermediates

2. For inner reduction / multi-stage:
   a) reduction_dim <= 64: threads_per_row = 8-16, multiple rows per block
   b) reduction_dim <= 3072: threads_per_row = 32, 4 rows per block (128 total)
   c) reduction_dim <= 16384: threads_per_row = 128, 1 row per block
   d) reduction_dim > 16384: threads_per_row = 256, 1 row per block, loop over chunks

3. Vector width:
   Always 128 / element_bits (8 for fp16, 4 for fp32)

4. Coalescing target:
   subgroup_size * vector_width (256 for NVIDIA fp16, 512 for AMD GFX9 fp16)

5. Shared memory:
   - For reductions: trivial (num_warps * sizeof(accum))
   - For transpose/conflicting coalescing: tile_size + padding
   - For data caching: hidden_dim * sizeof(T) if fits, else re-read from global
```

---

## 13. WHAT V2 SHOULD DO DIFFERENTLY FROM V1

Based on this analysis:

1. **Separate elementwise from reductions in the budget model.** Elementwise ops distribute work across all threads -- the budget is per-thread bytes, not per-WG total work. Reduction ops have sequential per-thread accumulation -- the budget bounds iterations per thread.

2. **Use the `threads_per_row` table** for lane_basis production. Don't derive it solely from coalescing -- use the battle-tested breakpoints at 64, 128, 3072, 6144, 16384.

3. **Always use 128-bit vector loads.** No need to consider smaller vector widths. If alignment doesn't permit 128 bits, fall back to scalar (like CUTLASS does with runtime alignment checks).

4. **Keep block sizes simple.** 128 for norms (matches Apex, FlashInfer, Quack), 256 for general reductions (matches CUB, PyTorch, TVM, MIOpen, CUTLASS), scale up only for very large reductions.

5. **Detect coalescing conflicts and promote to shared memory** rather than dropping constraints. Every library that handles transpose uses shared memory with bank-conflict padding. This is the correct solution.

6. **Multi-row workgroups for small reduction dims.** When hidden_dim < 128, pack 4-16 rows per workgroup (like Apex WARPS_M=4, Liger-Kernel BLOCK_ROW=16, Quack rows_per_block).

7. **Multi-block cooperation for very large reduction dims.** When hidden_dim > 16384, consider cooperative launch or cluster (like Apex CTAS_PER_ROW, Quack cluster_n). But this is rare for typical LLM hidden dims (4096-8192).

8. **The num_warps heuristic from Liger-Kernel is a good starting point:**
   ```
   BLOCK_SIZE < 2048:  num_warps = 4  (128 threads)
   BLOCK_SIZE < 8192:  num_warps = 8  (256 threads)
   BLOCK_SIZE < 32768: num_warps = 16 (512 threads)
   BLOCK_SIZE >= 32768: num_warps = 32 (1024 threads)
   ```

9. **CUB's `MemBoundScaling` should inform how tile sizes scale with element type.** The formula `items = nominal * 4 / sizeof(T)` keeps bytes-per-tile constant. This is the right approach for the budget.

10. **Don't autotune for memory-bound ops.** None of the Triton libraries actually use autotune for norms/softmax (commented-out in Liger, Unsloth). The configuration space is small enough for heuristics. CUB tunes empirically per GPU generation but publishes fixed constants.
