# V1 Prototype Shortcomings

What the v1 prototype doesn't do well, assessed against the 52 library kernel
benchmarks and the cross-cutting lessons from 12 kernel libraries.

---

## 1. Budget model conflates parallel and sequential work

**The problem:** The budget formula is `product(all_tiles) * maxElemBits`. For a
reduction op this is reasonable — each thread accumulates sequentially. But for
an elementwise op with tiles P0=256, P1=512, the 131K elements are *distributed*
across 64 threads (~2K each), not 131K sequential ops. The budget treats them
identically and over-counts elementwise work by 64x.

**Concrete failure:** Example 10 (elementwise transpose, 1024x4096 f16->f32).
Both input and output want coalescing along different dims. The budget panics
and drops *both* constraints, leaving zero coalescing. The v1 spec acknowledges
this as a known issue.

**What libraries do:** CUB's `MemBoundScaling` works in bytes-per-thread, not
total-per-WG. PyTorch's elementwise kernel has `block_work_size = num_threads *
thread_work_size` — per-thread work is independent, no aggregate budget concept.
Liger-Kernel DyT tiles both dimensions independently for elementwise.

**What it costs:** Every pure elementwise kernel (silu, swiglu, geglu, broadcast
add, mixed-precision dequant, large 1D, tiny batch) gets an over-conservative
budget that may force unnecessary constraint dropping or prevent multi-dim
coalescing.

**Affected kernels:** `silu_elementwise`, `swiglu_fused`, `geglu_fused`,
`elementwise_transpose`, `elementwise_broadcast`, `elementwise_large_1d`,
`elementwise_dynamic`, `elementwise_mixed_precision`, `elementwise_3d_broadcast`,
`elementwise_tiny_batch`, `elementwise_unaligned`.

---

## 2. No shared memory promotion for coalescing conflicts

**The problem:** When input and output want coalescing along different dimensions
(transpose), the only strategy v1 has is to *drop* one or both constraints. The
correct solution — shared memory promotion with bank-conflict padding — is never
considered.

**Concrete failure:** `transpose_pure_2d` (2048x4096 f16 -> 4096x2048) needs a
32x32 shared memory tile with +1 padding. V1 would drop coalescing on one side,
leaving strided accesses that waste 75-97% of each cache line.

**What libraries do:** Every library that handles transpose uses shared memory:
CUTLASS `OutputTileOptimalThreadMap`, TVM explicit `Transpose` rule with shmem +
bank-conflict padding, ThunderKittens swizzle patterns, PyTorch
`cunn_SoftMaxForward` vs spatial softmax split. IREE's own `setTransposeConfig`
already does this with 32x32 tiles — v1 just doesn't invoke it.

**Affected kernels:** `elementwise_transpose`, `transpose_pure_2d`,
`transpose_3d_inner`, `layernorm_transpose_output`, `softmax_transpose_output`,
`rms_norm_transpose_input`.

**Scoping note:** We don't need to handle shared memory promotion explicitly in
the config — specifying output coalescing should be sufficient, and the
downstream pipeline can insert promotion as needed. However, if we are
coalescing the output, we want to write multiple rows at once rather than one
row at a time (shared memory incurs a barrier, so amortizing it over multiple
rows is important). The tile size calculation should account for this multi-row
write batching.

---

## 3. No lane_basis / subgroup_basis emission

**The problem:** V1 analyzes parallelism, computes workgroup tiles, determines
per-op tiles — then returns `failure()`. It never produces the `lane_basis`,
`subgroup_basis`, or any other lowering config attributes. The existing
`setReductionConfig` handles all actual code generation.

**What this means:** The analysis is purely diagnostic. None of the insights
about multi-op tensor dimension tracking, per-operand coalescing targets, or
budget-aware constraint dropping actually influence code generation. The gap
between "analysis that knows what's right" and "config that drives codegen" is
the entire gap.

**What's needed:** The `coalesced_operands` annotation that `ConfigureTensorLayouts`
would consume to derive thread distribution. The Quack/FlashInfer
`threads_per_row` table provides battle-tested breakpoints for this mapping.

---

## 4. Single subgroup per workgroup only

**The problem:** V1 hardcodes `subgroupsPerWG = 1`. Real kernels use 4-32
warps per block depending on the operation.

**What libraries do:**
- Norms: 4 warps (128 threads) for hidden < 2048, 8 warps (256) standard,
  16 warps (512) for large hidden
- Softmax large vocab: 32 warps (1024 threads)
- CUB reduce: 8 warps (256 threads) default
- Liger-Kernel: `num_warps = {4, 8, 16, 32}` based on BLOCK_SIZE

**What it costs:** With 1 subgroup (64 threads on AMD, 32 on NVIDIA), v1 can't
express the standard 256-thread block that every library converges on for
general reductions. Cross-warp reduction via shared memory — the universal
Level 3 pattern — is impossible with a single subgroup.

**Affected kernels:** Nearly all. The `softmax_large_vocab` (1024 threads),
`rms_norm_very_large` (256+ threads), `cross_entropy_fused` (512 threads), and
`global_reduction` (256 threads) are most impacted.

**Scoping note:** The driver for multi-subgroup is occupancy: when there aren't
enough workgroups launched to fill the GPU, we should increase the number of
subgroups per workgroup. This is a reactive scaling decision (insufficient WG
parallelism -> add subgroups), not a fixed choice per operation class.

---

## 5. No multi-row workgroups for small reduction dimensions

**The problem:** When the reduction dimension is small (< 128 elements), one row
doesn't need a full warp. The right strategy is to pack multiple rows into one
workgroup so threads aren't wasted. V1 always assigns one row per workgroup
(wg_tile = 1 for parallel groups by default, then coalescing may increase it,
but never for the purpose of multi-row packing).

**What libraries do:**
- Apex: `WARPS_M=4` for hidden <= 2304 (4 rows per block)
- Liger-Kernel: `BLOCK_ROW=16` for BLOCK_SIZE <= 256
- Quack/FlashInfer: `rows_per_block = total_threads / threads_per_row`
  (e.g., N<=64 -> 8 threads/row -> 32 rows/block with 256 threads)
- `layernorm_small_hidden` (32768x64 f32): optimal is 8 rows/block with
  32 threads per row

**Affected kernels:** `rms_norm_small_hidden` (8192x128), `layernorm_small_hidden`
(32768x64), `elementwise_unaligned` (1000x3), `softmax_small` (512x128).

---

## 6. No multi-block cooperation for very large reductions

**The problem:** When the reduction dimension is very large (> 16384), a single
workgroup has low occupancy because there aren't enough parallel rows to fill
the GPU. V1 has no mechanism for splitting one reduction across multiple
workgroups with a global synchronization step.

**What libraries do:**
- Apex: `CTAS_PER_ROW=2,4,8` with cooperative launch + global memory barriers
- Quack: CTA clusters with distributed shared memory
- CUB: Multi-block reduce with `GridEvenShare` partitioning + 2-pass finalize
- `global_reduction` (16M f32): needs ~2160 blocks with 2-pass pattern
- `rms_norm_very_large` (64x65536): only 64 WGs, most SMs idle

**Affected kernels:** `rms_norm_very_large` (65536 reduction),
`softmax_large_vocab` (32000), `softmax_very_large_vocab` (128256),
`inner_reduction_f16_large` (65536), `global_reduction` (16M scalar),
`cross_entropy_fused` (32000).

**Out of scope:** We will not handle multi-block cooperation. If this is needed,
the input IR should already be structured with the multi-block split before
reaching our config pass. This is a dispatch formation concern, not a config
concern.

---

## 7. No 2D thread block structure for outer reductions

**The problem:** Outer reductions (reduce along a non-contiguous dimension) need
a 2D thread block: `threadIdx.x` for the contiguous/parallel dimension
(coalescing) and `threadIdx.y` for cooperative reduction along the strided
dimension. V1's coalescing walk identifies the right minimum tiles for the
parallel dimension, but has no mechanism to express a 2D thread distribution
where some threads cooperate on reduction.

**What's optimal:** `outer_reduction` (4096x1024 f32): block (32,16)=512
threads, threadIdx.x->d1 (coalesced), threadIdx.y->d0 (reduction cooperation).
`matvec_transposed` (4096x4096 f16): block (32,8)=256 threads, same pattern.
`softmax_outer_dim` (4096x1024 f32): block (32,8)=256 threads.

**What v1 does:** Identifies that P0 needs a minimum WG tile for coalescing but
can't express the cooperative reduction dimension in the thread mapping.

**Affected kernels:** `outer_reduction`, `outer_reduction_f16`,
`outer_reduction_small_parallel`, `outer_reduction_3d`, `outer_max_reduction`,
`outer_reduction_welford`, `outer_reduction_dynamic`, `softmax_outer_dim`,
`matvec_transposed`, `rms_norm_transpose_input`, `reduction_3d`.

**Not actually a problem:** Choosing the right workgroup tile sizes automatically
produces a 2D thread block — the downstream pipeline derives the thread
structure from the tile sizes. There's no need to explicitly construct 2D thread
blocks in the config; the right tile sizes are sufficient.

---

## 8. Broadcast cost computed but never used

**The problem:** Phase 3 computes broadcast cost per parallel group — how much
bandwidth is wasted by tiling a group that some operands don't index into. This
is exactly the information needed for coarsening decisions (which group to tile
less aggressively). But Phase 4 doesn't use it. The coarsening algorithm was
removed as "not well thought out."

**What this means:** The analysis knows which groups are expensive to tile (high
broadcast cost) but doesn't act on it. In cases where the GPU has more
parallelism than needed, there's no principled way to decide which groups to
coarsen.

**Where it matters:** Fused dispatches with broadcast operands.
`elementwise_broadcast` (2048x4096 + 4096): tiling P0 broadcasts the bias
vector. `elementwise_3d_broadcast` (8x512x4096 + 4096): tiling batch or seq
dims broadcasts the bias. `fused_residual_rms_norm`: norm weights broadcast
across batch.

---

## 9. No handling of non-standard reduction combiners

**The problem:** V1's analysis identifies reduction dimensions and parallelizable
groups, but doesn't reason about the reduction combiner type. Standard
add-reductions, max-reductions, and Welford (mean+variance) reductions have
different per-thread state requirements and different cross-thread merge
operations.

**What libraries do:**
- Argmax carries (value, index) pairs with comparison-based merge
- Welford carries (count, mean, m2) triples with the Welford parallel merge formula
- Cross-entropy uses online logsumexp with (max, sum_exp) pairs

**What it costs:** The budget check uses `maxElemBits` per element, but Welford
needs 3x the register state per thread. Argmax needs 2x. The budget
under-counts register pressure for these operations.

**Affected kernels:** `argmax_row`, `outer_reduction_welford`, `cross_entropy_fused`.

**Deferred:** We will only handle single reduction combiners initially. Multi-
combiner support (Welford, argmax) depends on vectorization being able to handle
these patterns. Can be added later when that infrastructure is in place.

---

## 10. No vectorization-aware handling of unaligned dimensions

**The problem:** V1 rounds tile sizes to powers of 2 and relies on masking for
out-of-bounds lanes. But it doesn't consider whether the innermost dimension
is too small or misaligned for vectorization.

**Concrete cases:**
- `elementwise_unaligned` (1000x3 f32): innermost dim is 3, not divisible by
  any useful vector width. V1 would still set a coalescing target of 256 (64*4)
  along dim 1, which is impossible with only 3 elements. The right strategy is
  to linearize to 1D (3000 elements) and use flat scalar access.
- `row_sum_unaligned` (1000x511 f16): dim 1 = 511, not divisible by 8 (f16
  vec width). Need masking on the last vector load per row.
- `layernorm_unaligned_3d` (16x100x511 f16): prime-ish reduction dim (511).

**What libraries do:** CUTLASS falls back to scalar loads when alignment doesn't
permit vectorization. CUB adjusts `VECTOR_LOAD_LENGTH` based on alignment.
PyTorch's vectorized_inner_reduction checks alignment at runtime.

**Resolution approach:** When the innermost dimension is too small for full
coalescing, span multiple contiguous dimensions to reach the coalescing target.
For `1000x3`, coalescing needs both the 3 and part of the 1000 dimension. The
implementation would look like `collapse_shape -> coalesced_masked_load ->
shape_cast`: collapse the contiguous dims, do a power-of-2 vector load with
masking, then reshape back. Tile sizes should still be powers of 2 — the
unaligned handling is an implementation detail inside the coalescing lowering,
not the tile size selection.

---

## 11. No 1D linearization for pure elementwise ops

**The problem:** Pure elementwise operations on multi-dimensional tensors don't
need 2D workgroup tiling. Linearizing to 1D gives simpler scheduling, better
occupancy, and trivial coalescing. V1 treats all dimensions as separate groups
and applies multi-dimensional tiling uniformly.

**What's optimal:**
- `elementwise_large_1d` (16M f16): 1D, 128 threads, vec8, 16384 WGs
- `elementwise_tiny_batch` (1x65536 f16): batch dim is a no-op, treat as 1D
- `silu_elementwise` (2048x4096 f16): can linearize to 8M elements

**What libraries do:** PyTorch's elementwise kernel always linearizes to 1D.
CUB operates on flat ranges. Only when there's a broadcast operand does 2D
structure matter.

**Not actually a problem (given constraint):** Kernel formation already
linearizes dimensions when possible. If dimensions remain separate in the input
IR, it's because they *must* be — e.g., one was transposed somewhere, or they
have different roles in different ops within the dispatch. The dimension
structure in the IR is a given; the config should respect it rather than trying
to re-linearize.

---

## 12. No consideration of L2 cache reuse patterns

**The problem:** V1 reasons about coalescing (L1/cache-line level) but not about
L2 reuse. Some operands are small enough to live in L2 across workgroups.

**Concrete cases:**
- `matvec_f16`: vector x (4096 * 2 = 8KB) fits in L2, broadcast across all
  4096 workgroups. No need to coalesce x — it's a broadcast.
- `rope_embedding`: sin/cos tables (2048*64*2 = 256KB) reused across batch and
  heads. Fits in L2 after first access.
- Norm weights/biases: typically 4096-16384 elements, always in L2.

**What it costs:** V1 may waste coalescing budget on operands that are already
cached, constraining tiles for the operands that actually need coalescing.

---

## 13. Row-major layout assumption baked in

**The problem:** The coalescing walk assumes innermost dim = last dim = contiguous
in memory. This is hardcoded throughout the algorithm. The v1 spec acknowledges
this as a known limitation.

**Where it breaks:** Any tensor with a non-standard memory layout (tiled layouts,
channel-last NHWC with non-trivial strides, or operands accessed through
non-identity indexing maps where the "last dim" in the indexing map isn't the
last dim in memory).

**Affected kernels:** `groupnorm_nhwc` (NHWC layout where W is innermost but
the reduction spans C, H, W), `transpose_4d_bsnh_to_bnsh` (innermost dim
preserved but outer dims permuted).

**Resolution approach:** The input/output boundaries carry stride information
that should be used when determining coalescing. Rather than assuming row-major
from dimension ordering alone, the coalescing analysis should consult the
strides at dispatch tensor load/store boundaries to determine actual memory
contiguity.

---

## Summary table

| # | Shortcoming | Status | Severity | Affected kernel count |
|---|------------|--------|----------|----------------------|
| 1 | Budget conflates parallel/sequential work | **Fix in v2** | High | 11+ (all elementwise) |
| 2 | No shared memory promotion for coalescing conflicts | **Scoped** — specify output coalescing + multi-row write batching | High | 6 (all transpose variants) |
| 3 | No lowering config emission | **Fix in v2** | Critical | All 52 |
| 4 | Single subgroup only | **Fix in v2** — scale subgroups when WG count is insufficient | High | ~40 (anything needing >64 threads) |
| 5 | No multi-row workgroups | **Fix in v2** | Medium | 4 (small reduction dims) |
| 6 | No multi-block cooperation | **Out of scope** — dispatch formation concern | N/A | N/A |
| 7 | No 2D thread blocks for outer reductions | **Not a problem** — tile sizes produce this automatically | N/A | N/A |
| 8 | Broadcast cost unused | **Fix in v2** | Low | 3+ (fused broadcasts) |
| 9 | No non-standard combiner handling | **Deferred** — needs vectorization support first | Medium | 3 (argmax, Welford, cross-entropy) |
| 10 | No unaligned vectorization handling | **Fix in v2** — multi-dim collapse for coalescing | Medium | 3+ (small/odd innermost dims) |
| 11 | No 1D linearization for elementwise | **Not a problem** — kernel formation already handles this | N/A | N/A |
| 12 | No L2 cache reuse reasoning | **Low priority** | Low | 3+ (broadcast/small operands) |
| 13 | Row-major layout assumption | **Fix in v2** — use stride info from dispatch boundaries | Medium | 2+ (non-standard layouts) |
