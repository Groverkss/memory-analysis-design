# V1 Prototype: Criticisms and Shortcomings

Assessed against the 52 library kernel benchmarks and cross-cutting lessons
from 12 kernel libraries.

---

## Critical

### 1. No lowering config emission

The prototype returns `failure()` after analysis -- it doesn't actually produce
lowering configs. No `lane_basis`, `subgroup_basis`, or any other attributes.
The existing `setReductionConfig` handles all actual code generation.

The analysis is purely diagnostic. None of the insights about multi-op tensor
dimension tracking, per-operand coalescing targets, or budget-aware constraint
dropping actually influence code generation.

What's needed: a `coalesced_operands` annotation that `ConfigureTensorLayouts`
would consume to derive thread distribution. The Quack/FlashInfer
`threads_per_row` table provides battle-tested breakpoints for this mapping.

**Affects:** All 52 kernels.

### 2. Single subgroup per workgroup only

V1 hardcodes `subgroupsPerWG = 1`. Real kernels use 4-32 warps per block.
Library convergence: 128 threads for norms, 256 for general reductions,
512-1024 for large reductions. With 1 subgroup (64 threads on AMD), v1 can't
express the standard 256-thread block.

The driver for multi-subgroup is occupancy: when there aren't enough workgroups
to fill the GPU, increase subgroups per workgroup. This is a reactive scaling
decision, not a fixed per-operation choice.

**Affects:** ~40 kernels (anything needing >64 threads).

---

## High Severity

### 3. Budget model conflates parallel and sequential work

The budget formula is `product(all_tiles) * maxElemBits`. For a reduction op
this is reasonable -- each thread accumulates sequentially. But for an
elementwise op with tiles P0=256, P1=512, the 131K elements are distributed
across 64 threads (~2K each), not 131K sequential ops. The budget over-counts
elementwise work by ~64x.

Concrete failure: Example 10 (elementwise transpose, 1024x4096 f16->f32).
Both input and output want coalescing along different dims. The budget panics
and drops both constraints, leaving zero coalescing.

Libraries solve this differently: CUB's `MemBoundScaling` works in
bytes-per-thread. PyTorch elementwise has `block_work_size = num_threads *
thread_work_size` (per-thread, independent).

**Affects:** 11+ kernels (all elementwise).

### 4. No shared memory promotion for coalescing conflicts

When input and output want coalescing along different dimensions (transpose),
the only strategy v1 has is to drop one or both constraints. The correct
solution -- shared memory promotion with bank-conflict padding -- is never
considered.

Every library that handles transpose uses shared memory: CUTLASS
`OutputTileOptimalThreadMap`, TVM explicit `Transpose` rule, ThunderKittens
swizzle patterns, IREE's own `setTransposeConfig` with 32x32 tiles.

Scoping note: we don't need to handle shared memory promotion explicitly in
the config -- specifying output coalescing should be sufficient, and the
downstream pipeline can insert promotion as needed. However, the tile size
calculation should account for multi-row write batching (shared memory incurs
a barrier, so amortizing over multiple rows matters).

**Affects:** 6 kernels (all transpose variants).

### 5. No multi-row workgroups for small reduction dimensions

When the reduction dimension is small (< 128 elements), one row doesn't need
a full warp. The right strategy is to pack multiple rows into one workgroup.
V1 always assigns one row per workgroup by default.

Libraries: Apex `WARPS_M=4` for hidden <= 2304. Liger-Kernel `BLOCK_ROW=16`
for BLOCK_SIZE <= 256. Quack/FlashInfer `rows_per_block = total_threads /
threads_per_row`.

**Affects:** 4 kernels (small reduction dims).

---

## Medium Severity

### 6. No 2D thread block structure for outer reductions

Outer reductions need a 2D thread block: `threadIdx.x` for the contiguous
dimension (coalescing) and `threadIdx.y` for cooperative reduction along the
strided dimension. V1's coalescing walk identifies the right minimum tiles but
can't express a 2D thread distribution.

Not actually a problem in practice: choosing the right workgroup tile sizes
automatically produces a 2D thread block -- the downstream pipeline derives
the thread structure from the tile sizes. The right tile sizes are sufficient.

**Affects:** 11 kernels (outer reductions, transposed matvec).

### 7. No non-standard reduction combiners

V1 doesn't reason about combiner type. Argmax carries (value, index) pairs.
Welford carries (count, mean, m2) triples. Cross-entropy uses online
logsumexp with (max, sum_exp) pairs. The budget under-counts register pressure
for these.

Deferred: only handle single reduction combiners initially. Multi-combiner
depends on vectorization infrastructure.

**Affects:** 3 kernels (argmax, Welford, cross-entropy).

### 8. No vectorization-aware handling of unaligned dimensions

V1 rounds tiles to powers of 2 and relies on masking, but doesn't consider
whether the innermost dimension is too small or misaligned for vectorization.
`elementwise_unaligned` (1000x3 f32) has innermost dim 3 -- can't vectorize.
The right strategy is to span multiple contiguous dimensions to reach the
coalescing target.

**Affects:** 3+ kernels (small/odd innermost dims).

### 9. Row-major layout assumption baked in

The coalescing walk assumes innermost dim = last dim = contiguous in memory.
Hardcoded throughout. Breaks for non-standard memory layouts (NHWC with
non-trivial strides, tiled layouts). Should consult strides at dispatch tensor
load/store boundaries.

**Affects:** 2+ kernels (non-standard layouts).

---

## Low Severity

### 10. Broadcast cost computed but unused

Phase 3 computes broadcast costs per parallel group but Phase 4 doesn't use it
for coarsening. The coarsening algorithm was removed as "not well thought out."
The analysis knows which groups are expensive to tile but doesn't act on it.

**Affects:** 3+ kernels (fused broadcasts).

### 11. No L2 cache reuse reasoning

V1 reasons about coalescing (L1/cache-line level) but not L2 reuse. Some
operands are small enough to live in L2 across workgroups (matvec's vector x
at 8KB, norm weights at 4096 elements). May waste coalescing budget on
already-cached operands.

**Affects:** 3+ kernels (broadcast/small operands).

---

## Out of Scope

### 12. No multi-block cooperation for very large reductions

When the reduction dim is very large (> 16384), a single workgroup has low
occupancy. V1 has no mechanism for splitting one reduction across multiple
workgroups with global synchronization.

This is a dispatch formation concern, not a config concern. If needed, the
input IR should already be structured with the multi-block split.

### 13. No 1D linearization for pure elementwise ops

Kernel formation already linearizes dimensions when possible. If dimensions
remain separate in the input IR, it's because they must be. The config should
respect the dimension structure, not try to re-linearize.

---

## Key Reviewer Feedback from Design Iterations

These inline corrections shaped the design but are worth preserving:

1. **"Don't pick a single dominant operand."** Multiple operands can need
   coalescing. If layouts disagree, VectorDistribute automatically inserts
   shared memory conversions. Don't try to resolve conflicts in the config.

2. **"Always use max vector width."** No adjustment loop for divisibility.
   Masking handles non-divisible sizes. The old code over-complicated this.

3. **"Specifying coalescing per-operand is better."** Rather than picking one
   layout for the iteration space, specify coalescing per-operand and let
   ConfigureTensorLayouts handle conflicts with shared memory.

4. **"You can't intersect iteration space dims across ops."** Each op has its
   own iteration space. The connection is through tensor SSA values. This led
   to the tensor-centric approach.

5. **"Reduction dims are part of the contiguous region."** The coalescing walk
   should not stop at the first reduction dim. Reduction dims contribute to
   coalescing for free. Only stop at stride gaps.

6. **"Prioritize larger operand, not always input."** When input/output
   coalescing conflict, prioritize by total bytes, not by read-vs-write role.

7. **"Budget should be a fixed seed."** `subgroupSize * maxLoadBits * k`
   where k is a tunable constant (start with 8).

---

## Summary Table

| # | Shortcoming | Fix Status | Severity |
|---|------------|------------|----------|
| 1 | No lowering config emission | Fix in v2 | Critical |
| 2 | Single subgroup only | Fix in v2 | Critical |
| 3 | Budget conflates parallel/sequential | Fix in v2 | High |
| 4 | No shared memory promotion | Scoped for v2 | High |
| 5 | No multi-row workgroups | Fix in v2 | High |
| 6 | No 2D thread blocks (outer reduction) | Not a problem (tiles produce this) | Medium |
| 7 | No non-standard combiners | Deferred | Medium |
| 8 | No unaligned vectorization | Fix in v2 | Medium |
| 9 | Row-major assumption | Fix in v2 | Medium |
| 10 | Broadcast cost unused | Fix in v2 | Low |
| 11 | No L2 cache reasoning | Low priority | Low |
| 12 | No multi-block cooperation | Out of scope | N/A |
| 13 | No 1D linearization | Not a problem | N/A |
