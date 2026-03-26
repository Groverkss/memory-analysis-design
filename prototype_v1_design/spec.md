# Memory-Bound Config v1 Spec

## Goal

Replace the existing `setReductionConfig` with a unified configuration system
for memory-bound GPU operations (reductions, scans, sorts, elementwise, etc.)
in IREE's VectorDistribute pipeline. The system should handle multi-op fused
dispatches, determine thread distribution based on memory coalescing, and work
uniformly with dynamic shapes.

## Architecture

The implementation lives in `MemoryBoundConfigUtils.cpp` alongside the existing
`ReductionConfigUtils.cpp`. It's gated behind a flag
`--iree-codegen-llvmgpu-use-memory-bound-config` (default off) and hooks in
before `setReductionConfig` in `KernelConfig.cpp`, falling through on failure.

Currently the analysis runs and prints debug output but does NOT produce
lowering configs -- it returns `failure()` so the existing pipeline takes over.

## Core Principle

**Thread distribution is driven by memory coalescing, not by whether a
dimension is parallel or reduction.**

For coalesced global memory access, consecutive threads must access
consecutive memory addresses. Threads should be distributed along the
**contiguous (innermost) dimension of the dominant operand**.

Whether that dimension is a reduction dim or a parallel dim determines
the *reduction strategy*, not the *thread layout*:
- If threads are along a **reduction** dim -> cooperative reduction (shuffles)
- If threads are along a **parallel** dim -> independent work, reduction is
  serial per thread (loop accumulation)

Both strategies are already supported by the existing VectorDistribute pipeline.

## Algorithm: Overall Flow

```
analysis = analyzeParallelism(entryPoint)
seeds = buildSeeds(target)
droppedTensors = {}
loop (max 16 iterations):
  tiling = computeWorkgroupTiling(analysis, seeds, droppedTensors)
  perOpTiles = computePerOpTiles(analysis, tiling, seeds)
  tensorToDrop = budgetCheck(analysis, tiling, perOpTiles, seeds)
  if tensorToDrop: droppedTensors.add(tensorToDrop); continue
  else: break
```

### Stage 1: Parallelism Analysis

Input: All linalg ops in the dispatch (gathered via backward slice from stores).

Steps:
1. **Gather compute ops** -- backward slice from `DispatchTensorStoreOp` /
   `StoreToBufferOp`. Excludes `FillOp`.
2. **Build tensor infos** -- for each tensor value, record which ops access
   it, through which indexing maps, and whether each tensor dim is
   reduction in any accessing op.
3. **Build dimension groups** -- union-find: for each op, link tensor dims
   that map to the same iteration dim. Op results are included via
   `getIndexingMapMatchingResult()`. A group is parallel only if ALL
   members are parallel in every op.
4. **Resolve dynamic dims** -- `TensorDynamicDimAnalysis` + `IntegerRangeAnalysis`
   for upper bounds. Default 1M for truly unknown dims.

Output: `ParallelismAnalysis` with `computeOps`, `tensorInfos`, `groups`
(each with `isParallelizable`, `size`, `resolvedSize`).

### Stage 2: Workgroup Tiling

Input: Parallelism analysis + HeuristicSeeds + droppedTensors set.

**Phase 1 -- Coalescing Constraints:**
For each tensor that accesses global memory (dispatch inputs + stored outputs),
skipping tensors in `droppedTensors`:
- Compute per-operand `coalescingTarget = subgroupSize * (maxLoadBits / elemBits)`
- Walk dims innermost-to-outermost:
  - Reduction dims: subtract from remaining target (free coalescing)
  - Parallel dims: set `groupMinTiles[group] = PowerOf2Ceil(min(dimSize, remaining))`
  - Stop when target met OR when a parallel dim tile < full size (stride gap)
- Take max across all tensors for each group's WG tile.

**Phase 2 -- Budget Check:**
Handled in the outer loop (see below).

**Phase 3 -- Dimension Interactions:**
For each parallel group, compute broadcast cost: total elements of operands
that do NOT have a dim in this group. Higher broadcast cost = tiling this
group wastes more bandwidth.

**Phase 4 -- Workgroup Count:**
`numWGs = product(resolvedSize / wgTile)` for all parallel groups.
1 subgroup for now. No coarsening or subgroup scaling.

**Phase 5 -- Suggestions:**
Warn on low occupancy. Warn when coalescing constraints consume most of a
group's parallelism.

Output: `WorkgroupTiling` with `groupTiles`, `numWGs`, `subgroupsPerWG`,
and analysis details for debug output.

### Stage 3: Per-Op Tile Sizes

Input: Parallelism analysis + WorkgroupTiling + HeuristicSeeds.

For each compute op in topological order:
1. **Initialize tiles:** Parallel dims get WG tile, reduction dims start at 1.
2. **Apply coalescing from dispatch inputs:** Walk innermost-to-outermost,
   increase reduction tiles to PowerOf2Ceil of coalescing target.
   Stop at stride gaps (tile < full dim size).
3. **Propagate reduction tiles:** For each reduction group, take the max tile
   across all ops and apply to all. This ensures consistency -- if op A gets
   R0=512 from coalescing, downstream op B sharing R0 also gets 512.

Output: `PerOpTileResult` with per-op `minTiles` and `iterToGroup` mapping.

### Stage 4: Budget Check

Input: All analysis results + HeuristicSeeds.

Budget: `subgroupSize * maxLoadBits * maxVectorLoadsPerThread` (in bits).
Default: 64 * 128 * 8 = 65,536 bits.

For each compute op: `perWGWorkBits = product(tiles) * maxElemBits(op)`.

If any op exceeds budget:
1. Find the worst op (highest work in bits).
2. Score each coalescing constraint by how much dropping it would reduce the
   worst op's work. Score = product of (currentTile / tileWithoutConstraint)
   for each parallel group the constraint sets.
3. Drop the highest-scoring constraint (remove its tensor from future
   coalescing analysis).
4. Recompute everything and check again.

If no droppable constraint helps (e.g., over-budget is purely from reduction
dims), terminate -- accept the situation.

## Data Structures

```
TensorDim { Value tensor; int64_t dim; }
TensorAccess { LinalgOp op; OpOperand *operand; AffineMap indexingMap; }
TensorInfo { Value tensor; RankedTensorType type;
             SmallVector<SmallVector<TensorAccess>> accessesPerDim;
             SmallVector<bool> isParallelizable; }
DimensionGroup { SmallVector<TensorDim> members; bool isParallelizable;
                 int64_t size; int64_t resolvedSize; }
ParallelismAnalysis { SetVector<LinalgOp> computeOps;
                      DenseMap<Value, TensorInfo> tensorInfos;
                      SmallVector<DimensionGroup> groups; }
HeuristicSeeds { coalescingWidthElements, significantOperandElements,
                 targetOccupancy, maxSubgroupsPerWG, subgroupSize,
                 maxLoadBits, maxVectorLoadsPerThread }
CoalescingConstraint { Value tensor; int64_t coalescingTarget, vectorWidth;
                       SmallVector<pair<int, int64_t>> groupMinTiles;
                       int64_t reductionContrib; bool satisfiedByReduction; }
GroupInteraction { int groupIdx; int64_t broadcastElements, affectedElements;
                   bool isCoalescingCritical; }
WorkgroupTiling { SmallVector<int64_t> groupTiles; int64_t subgroupsPerWG, numWGs;
                  SmallVector<CoalescingConstraint> coalescingConstraints;
                  SmallVector<GroupInteraction> groupInteractions;
                  SmallVector<string> suggestions; }
PerOpTileResult { DenseMap<Operation*, SmallVector<int64_t>> minTiles;
                  DenseMap<Operation*, SmallVector<int>> iterToGroup; }
```

## Test Examples (build/research/examples/)

| # | Case | Groups | Key behavior |
|---|------|--------|-------------|
| 01 | Innermost reduction 1024x4096 f16 | P0=1024, R0=4096 | Budget drops store coalescing. R0 tile=512. |
| 02 | Outermost reduction 4096x1024 f16 | P0=1024, R0=4096 | Both input+store need P0. Can't drop. 2 WGs. |
| 03 | Fused layernorm 2x32x10x16384 f16->f32 | P0=2,P1=32,R0=10,R1=16384 | R1 satisfies all coalescing. R0 tile=512 propagated. |
| 04 | Softmax 512x10240 f32 | P0=512, R0=10240 | R0 satisfies coalescing. R0 tile=256. |
| 05 | Dynamic reduction ?x? f16 | P0=1M, R0=1M | Same as 01 with resolved upper bounds. |
| 06 | Outermost odd 4096x511 f16 | P0=511, R0=4096 | P0 tile=512 (exceeds 511, masking). |
| 07 | Layernorm transposed output | P0=2,P1=32,R0=10,R1=16384 | Budget drops output coalescing. 64 WGs. |
| 08 | Layernorm transposed dynamic | All 1M | Budget drops output coalescing. |
| 09 | Pure elementwise 1024x4096 f16->f32 | P0=1024, P1=4096 | P1=512 from f16 input. Both coalesced. |
| 10 | Elementwise transpose f16->f32 | P0=1024, P1=4096 | Coalescing conflict. Budget drops both. **Known issue.** |

## Files

- `compiler/src/iree/compiler/Codegen/Dialect/GPU/TargetUtils/MemoryBoundConfigUtils.cpp` -- main impl (~1700 lines)
- `compiler/src/iree/compiler/Codegen/Dialect/GPU/TargetUtils/ConfigUtils.h` -- `setMemoryBoundConfig` declaration
- `compiler/src/iree/compiler/Codegen/LLVMGPU/KernelConfig.cpp` -- flag + hook
- `compiler/src/iree/compiler/Codegen/Dialect/GPU/TargetUtils/BUILD.bazel` -- build
- `compiler/src/iree/compiler/Codegen/Dialect/GPU/TargetUtils/CMakeLists.txt` -- build
- `build/research/examples/` -- 10 test IR files
- `build/research/prototype_v1_design/` -- design docs
- `build/research/background/` -- 7 background research docs

## Branch

`memory-bound-analysis` on the IREE repo.
