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
lowering configs — it returns `failure()` so the existing pipeline takes over.

## Key Design Decisions

### 1. Tensor-centric parallelism (not iteration-space-centric)

**What worked well:** Parallelism is analyzed as a property of tensor
dimensions, not iteration space dimensions. Each op has its own iteration
space; the connection between ops is through tensor SSA values. A union-find
groups (tensor, dim) pairs that must be tiled together across ops.

**Why this matters:** In a fused dispatch, `d0` in one op has no inherent
relationship to `d0` in another op. The existing `setReductionConfig` works
on a single op's iteration space and can't reason about multi-op dispatches.
Our analysis naturally handles any number of ops by linking through shared
tensors.

**Implementation:** `buildDimensionGroups()` uses `EquivalenceClasses` to
union tensor dims that map to the same iteration dim within any op. Results
are linked to init operands through `getIndexingMapMatchingResult()` — no
special DPS handling needed.

### 2. No dynamic shape special-casing

**What worked well:** All algorithm logic uses `resolvedSize` uniformly.
Static dims use their known size. Dynamic dims get upper bounds from
`IntegerRangeAnalysis` (via `TensorDynamicDimAnalysis`). If no bound is
available, a large default (1M) is used. There is zero branching on
"is this dim dynamic?" anywhere in the algorithm.

**Implementation:** After building dimension groups, a single pass resolves
all dynamic sizes. Every subsequent computation just reads `group.resolvedSize`.

### 3. Per-operand coalescing targets

**What worked well:** The coalescing target depends on the operand's element
type: `subgroupSize * (maxLoadBits / elementBits)`. An f16 operand needs 512
contiguous elements (64 threads * 8 elements/thread), while an f32 operand
needs only 256 (64 * 4). This correctly handles mixed-precision dispatches
like layernorm (f16 input, f32 accumulator/output).

**Previous bug:** v1 initially used a global `max(elementBits)` across all
tensors, giving the smallest (most conservative) target. This under-constrained
small-element operands. Fixed to be per-operand.

### 4. Coalescing walks contiguous dims, stops at stride gaps

**What worked well:** For each tensor, we walk dimensions from innermost
(contiguous in memory) to outermost. Reduction dims contribute to coalescing
for free (always fully within WG). Parallel dims require minimum WG tiles.

**Key correctness fix:** Once a dim's tile is less than its full size, there's
a stride gap between elements of the next outer dim. Outer dims are NOT
contiguous, so we stop accumulating. This correctly identifies that a
transposed output with inner parallel dims tiled to 1 has contiguous=1, not
contiguous=product(all tiles).

**Limitation:** We assume row-major layout (innermost dim = last dim). This
doesn't account for actual memory layouts.

### 5. Tile sizes are powers of 2, can exceed dim sizes

**What worked well:** All tile sizes are rounded to powers of 2 for clean
thread distribution. Tiles can exceed the dim size — masking handles
out-of-bounds lanes. E.g., dim size 511 gets tile 512 instead of being
capped at 511 (which isn't a power of 2 and doesn't achieve full coalescing).

### 6. HeuristicSeeds for tunable parameters

**What worked well:** All hardware-derived parameters are in a single struct
(`HeuristicSeeds`) constructed from `TargetAttr`. This makes it easy to
iterate on benchmarks without changing algorithm structure. Seeds include:
subgroupSize, maxLoadBits, targetOccupancy, maxSubgroupsPerWG,
maxVectorLoadsPerThread (budget).

### 7. Observability via debug output

**What worked well:** Three levels of debug output:
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
1. **Gather compute ops** — backward slice from `DispatchTensorStoreOp` /
   `StoreToBufferOp`. Excludes `FillOp`.
2. **Build tensor infos** — for each tensor value, record which ops access
   it, through which indexing maps, and whether each tensor dim is
   reduction in any accessing op.
3. **Build dimension groups** — union-find: for each op, link tensor dims
   that map to the same iteration dim. Op results are included via
   `getIndexingMapMatchingResult()`. A group is parallel only if ALL
   members are parallel in every op.
4. **Resolve dynamic dims** — `TensorDynamicDimAnalysis` + `IntegerRangeAnalysis`
   for upper bounds. Default 1M for truly unknown dims.

Output: `ParallelismAnalysis` with `computeOps`, `tensorInfos`, `groups`
(each with `isParallelizable`, `size`, `resolvedSize`).

### Stage 2: Workgroup Tiling

Input: Parallelism analysis + HeuristicSeeds + droppedTensors set.

**Phase 1 — Coalescing Constraints:**
For each tensor that accesses global memory (dispatch inputs + stored outputs),
skipping tensors in `droppedTensors`:
- Compute per-operand `coalescingTarget = subgroupSize * (maxLoadBits / elemBits)`
- Walk dims innermost-to-outermost:
  - Reduction dims: subtract from remaining target (free coalescing)
  - Parallel dims: set `groupMinTiles[group] = PowerOf2Ceil(min(dimSize, remaining))`
  - Stop when target met OR when a parallel dim tile < full size (stride gap)
- Take max across all tensors for each group's WG tile.

**Phase 2 — Budget Check:**
Handled in the outer loop (see below).

**Phase 3 — Dimension Interactions:**
For each parallel group, compute broadcast cost: total elements of operands
that do NOT have a dim in this group. Higher broadcast cost = tiling this
group wastes more bandwidth.

**Phase 4 — Workgroup Count:**
`numWGs = product(resolvedSize / wgTile)` for all parallel groups.
1 subgroup for now. No coarsening or subgroup scaling.

**Phase 5 — Suggestions:**
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
   across all ops and apply to all. This ensures consistency — if op A gets
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
dims), terminate — accept the situation.

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
| 03 | Fused layernorm 2x32x10x16384 f16→f32 | P0=2,P1=32,R0=10,R1=16384 | R1 satisfies all coalescing. R0 tile=512 propagated. |
| 04 | Softmax 512x10240 f32 | P0=512, R0=10240 | R0 satisfies coalescing. R0 tile=256. |
| 05 | Dynamic reduction ?x? f16 | P0=1M, R0=1M | Same as 01 with resolved upper bounds. |
| 06 | Outermost odd 4096x511 f16 | P0=511, R0=4096 | P0 tile=512 (exceeds 511, masking). |
| 07 | Layernorm transposed output | P0=2,P1=32,R0=10,R1=16384 | Budget drops output coalescing. 64 WGs. |
| 08 | Layernorm transposed dynamic | All 1M | Budget drops output coalescing. |
| 09 | Pure elementwise 1024x4096 f16→f32 | P0=1024, P1=4096 | P1=512 from f16 input. Both coalesced. |
| 10 | Elementwise transpose f16→f32 | P0=1024, P1=4096 | Coalescing conflict. Budget drops both. **Known issue.** |

## Known Limitations / Issues

### 1. Elementwise transpose drops all coalescing (example 10)

When input and output have conflicting coalescing (transposed layout), the
budget check drops both constraints. The correct solution is to coalesce both
using shared memory promotion (`promote_operands`) — load input coalesced into
shared memory, then write output coalesced from shared memory. The existing
`setTransposeConfig` does this with 32x32 tiles.

**Root cause:** The budget measures per-WG work as `product(tiles) * elemBits`,
which conflates parallel work (which can be done by many threads simultaneously)
with sequential work per thread. For an elementwise op with P0=256, P1=512,
the 131K elements are spread across 64 threads (2K elements each), not 131K
sequential ops. The budget is too conservative for elementwise ops because it
doesn't account for the parallelism within the WG tile.

**What's needed:** The budget should account for how work is distributed across
threads. For reduction ops, the work IS sequential (each thread accumulates).
For elementwise ops, the work is parallel (threads partition the 2D tile).
Additionally, shared memory promotion should be considered as a strategy to
resolve coalescing conflicts rather than dropping constraints.

### 2. No lane_basis / subgroup_basis production

The v1 prototype does not produce `lane_basis` or `subgroup_basis` for the
lowering config. The plan was to add a `coalesced_operands` annotation that
`ConfigureTensorLayouts` would use to derive thread distribution. This hasn't
been implemented.

### 3. Single subgroup only

The prototype assumes 1 subgroup per WG. Multi-subgroup configurations
(cooperative reductions, subgroup scaling for occupancy) are not implemented.

### 4. Row-major layout assumption

The coalescing walk assumes innermost dim = last dim = contiguous in memory.
This doesn't account for actual memory layouts (e.g., tiled layouts).

### 5. No lowering config emission

The prototype returns `failure()` after analysis — it doesn't actually produce
lowering configs. The existing `setReductionConfig` handles all actual
configuration.

### 6. Broadcast cost computed but unused

Phase 3 computes broadcast costs per parallel group (how much bandwidth is
wasted by tiling this group), but Phase 4 doesn't use it for coarsening. The
coarsening algorithm was removed as not well thought out.

### 7. Phase 2 budget interacts poorly with shared memory

The budget check doesn't consider shared memory promotion as a strategy. When
coalescing constraints conflict (transpose), the budget drops constraints
rather than introducing promotion. A v2 should consider:
- Detect coalescing conflicts (different operands want different dims coalesced)
- For conflicts: promote the non-coalesced operand through shared memory
- Only drop constraints when promotion is also not viable (e.g., shared memory
  budget exceeded)

## Files

- `compiler/src/iree/compiler/Codegen/Dialect/GPU/TargetUtils/MemoryBoundConfigUtils.cpp` — main impl (~1700 lines)
- `compiler/src/iree/compiler/Codegen/Dialect/GPU/TargetUtils/ConfigUtils.h` — `setMemoryBoundConfig` declaration
- `compiler/src/iree/compiler/Codegen/LLVMGPU/KernelConfig.cpp` — flag + hook
- `compiler/src/iree/compiler/Codegen/Dialect/GPU/TargetUtils/BUILD.bazel` — build
- `compiler/src/iree/compiler/Codegen/Dialect/GPU/TargetUtils/CMakeLists.txt` — build
- `build/research/examples/` — 10 test IR files
- `build/research/prototype_v1_design/` — 14 design iteration docs
- `build/research/background/` — 7 background research docs

## Branch

`memory-bound-analysis` on the IREE repo.
