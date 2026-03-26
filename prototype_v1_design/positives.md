# V1 Prototype: What Worked Well

## 1. Tensor-centric parallelism (not iteration-space-centric)

Parallelism is analyzed as a property of tensor dimensions, not iteration space
dimensions. Each op has its own iteration space; the connection between ops is
through tensor SSA values. A union-find groups (tensor, dim) pairs that must be
tiled together across ops.

Why this matters: In a fused dispatch, `d0` in one op has no inherent
relationship to `d0` in another op. The existing `setReductionConfig` works on
a single op's iteration space and can't reason about multi-op dispatches. Our
analysis naturally handles any number of ops by linking through shared tensors.

Implementation: `buildDimensionGroups()` uses `EquivalenceClasses` to union
tensor dims that map to the same iteration dim within any op. Results are linked
to init operands through `getIndexingMapMatchingResult()`.

## 2. No dynamic shape special-casing

All algorithm logic uses `resolvedSize` uniformly. Static dims use their known
size. Dynamic dims get upper bounds from `IntegerRangeAnalysis` (via
`TensorDynamicDimAnalysis`). If no bound is available, a large default (1M) is
used. There is zero branching on "is this dim dynamic?" anywhere in the
algorithm.

After building dimension groups, a single pass resolves all dynamic sizes.
Every subsequent computation just reads `group.resolvedSize`.

## 3. Per-operand coalescing targets

The coalescing target depends on the operand's element type:
`subgroupSize * (maxLoadBits / elementBits)`. An f16 operand needs 512
contiguous elements (64 threads * 8 elements/thread), while an f32 operand
needs only 256 (64 * 4). This correctly handles mixed-precision dispatches
like layernorm (f16 input, f32 accumulator/output).

Previous bug fixed: v1 initially used a global `max(elementBits)` across all
tensors, giving the smallest (most conservative) target. Fixed to be
per-operand.

## 4. Coalescing walks contiguous dims, stops at stride gaps

For each tensor, we walk dimensions from innermost (contiguous in memory) to
outermost. Reduction dims contribute to coalescing for free (always fully
within WG). Parallel dims require minimum WG tiles.

Key correctness fix: once a dim's tile is less than its full size, there's a
stride gap between elements of the next outer dim. Outer dims are NOT
contiguous, so we stop accumulating. This correctly identifies that a
transposed output with inner parallel dims tiled to 1 has contiguous=1, not
contiguous=product(all tiles).

Important insight from design iteration: the contiguous region includes ALL
dimensions (parallel AND reduction). A reduction dim is contiguous in memory
just like any other dim. The coalescing walk should not stop at the first
reduction dim -- it should recognize that reduction dims contribute to the
contiguous region for free.

## 5. Tile sizes are powers of 2, can exceed dim sizes

All tile sizes are rounded to powers of 2 for clean thread distribution. Tiles
can exceed the dim size -- masking handles out-of-bounds lanes. E.g., dim size
511 gets tile 512 instead of being capped at 511 (which isn't a power of 2 and
doesn't achieve full coalescing).

## 6. HeuristicSeeds for tunable parameters

All hardware-derived parameters are in a single struct (`HeuristicSeeds`)
constructed from `TargetAttr`. This makes it easy to iterate on benchmarks
without changing algorithm structure. Seeds include: subgroupSize, maxLoadBits,
targetOccupancy, maxSubgroupsPerWG, maxVectorLoadsPerThread (budget).

## 7. Observability via debug output

Three levels of debug output:
1. **Parallelism Analysis:** Symbolic pseudo-IR DAG with dimension symbols
   (P0, P1, R0, R1), showing the dispatch structure with dimension group
   annotations and resolved sizes.
2. **Workgroup Tiling:** Per-operand coalescing analysis, dimension
   interactions (broadcast costs), tiling decision with WG count and
   occupancy, per-WG dispatch in the same pseudo-IR format, and diagnostic
   suggestions.
3. **Per-Op Tile Sizes:** Per iteration dim tiles with dimension group
   mapping, and coalescing satisfaction check for each global memory operand
   (showing contiguous element count vs target).

## 8. Budget check with iterative constraint dropping

The outer loop detects when per-WG work exceeds a budget and drops the
least-valuable coalescing constraint (by scoring which constraint removal
gives the biggest work reduction). This prevents pathological cases like a
transposed output forcing the WG to process the entire dispatch in one
workgroup.

## 9. Core principle: coalescing drives thread distribution

The fundamental insight -- that thread distribution should be driven by
memory coalescing, not by whether a dimension is parallel or reduction -- is
validated by every library in the kernel analysis (Apex, PyTorch, CUB, Triton,
FlashInfer, etc.). This principle correctly handles innermost reduction,
outermost reduction, and mixed cases with a single unified algorithm.

## 10. Broadcast cost analysis

Phase 3 correctly identifies which operands are "broadcast" across each
parallel group (the operand has no dim in that group, so tiling the group
doesn't reduce per-WG traffic for that operand). This is the right framework
for understanding matvec-like patterns where lhs is broadcast across N and
rhs is broadcast across M.

## 11. Multi-dim coalescing constraints

The coalescing constraint is on the product of WG tiles across contiguous
parallel dims, not on each individually. This allows flexible allocation:
for `tensor<P_H x P_W x P_C>` with target 512, any combination where
`tile[P_C] * tile[P_W] * tile[P_H] >= 512` works. The algorithm fills from
innermost outward (matching memory order).

## 12. Validated by coalescing benchmarks

The HIP coalescing benchmarks on gfx942 and gfx1201 confirm:
- 128-bit vector loads are the right target (all widths reach same peak BW)
- Coalescing depends on the address set, not thread-to-address mapping
- The `subgroupSize * vectorWidth` coalescing target formula is correct
- Triton's 128-bit per-thread cap is empirically validated
