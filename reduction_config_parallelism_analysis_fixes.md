# Fixing ReductionConfigUtils with ParallelismAnalysis

## The Core Problem

`ReductionConfigUtils.cpp` assumes that iteration dim **indices** are meaningful
across operations. It computes "shared parallel dims" by intersecting the
**index sets** of parallel dims across all ops. This is wrong because each op
has its own iteration space. Dim index 1 in op A and dim index 1 in op B are
unrelated — they only share the same number. The actual connection between ops
is through **tensor SSA values** and indexing maps.

ParallelismAnalysis solves this by tracing through tensors to build **dimension
groups** — sets of (tensor, dim) pairs that must be tiled together — and
labeling each group as parallel or reduction.

---

## New Strategy

### 1. Pre-checks

- Root op must have a reduction iterator.
- Single combiner check on all reduction ops.
- No convolutions, no chained reductions (reduced operand from a reduction producer).
- Same as current `checkDispatchForVectorDistribution`.

### 2. Run ParallelismAnalysis

```cpp
FailureOr<ParallelismAnalysis> analysis = analyzeParallelism(entryPoint);
```

Gives us dimension groups with `isParallelizable` and `resolvedSize`.

### 3. Determine global parameters from root op only

Compute `threadLoads`, `workgroupSize`, `subgroupSize` from the root reduction
op's iteration space. **Do not adjust threadLoads for other ops** — that's done
per-op in step 4.

Current code adjusts the global threadLoads downward via
`getThreadLoadsConstraint` for all ops (L695-705), which uses the wrong dim for
parallel-only ops and can kill vectorization for the root op. We fix this by
deferring per-op threadLoads to step 4.

`reductionSize` and `parallelSize` can optionally be computed from analysis
groups for correctness, but computing from root op is fine for now (they agree
in the common case).

### 4. Populate per-op configuration (reverse topological order)

Walk from the last op (closest to store) toward producers. For each op:

**Skip if it can infer distribution from its consumer.** This matches the
current `shouldAttachLoweringConfig` returning false — pure elementwise with
identity outputs, no split users, no new input dims. (Broadcasts and
reductions always need config.)

**For ops that need config:**

Use `analysis.buildIterToGroupMap(op)` to find which iter dims belong to which
groups. Only distribute threads on dims that are **not fully parallel** (i.e.,
map to a reduction group). Distribute on the **innermost such local dim** —
this matches what the current code does:
- Reduction ops: `reductionDims.back()` = innermost reduction dim
- Parallel-only ops needing config: `parallelDims.back()` (currently; with the
  analysis, this becomes the innermost dim mapping to a reduction group)

For the distribution dim, compute **per-op threadLoads**:
- Start from `largestLoadSizeInBits / bitWidth`
- Adjust so `dimSize % threadLoads == 0` and `(dimSize / threadLoads) % subgroupSize == 0`
- This replaces the broken global `getThreadLoadsConstraint` loop.

Then compute the same thread/subgroup distribution as today:
- `partialReductionSize = gcd(workgroupSize * threadLoads, dimSize)`
- Fit `subgroupBasis * threadBasis * threadLoads` into `partialReductionSize`
- Set `partial_reduction`, `thread`, `lane_basis`, `subgroup_basis` on that dim.

**Track minWgTiles:** Initialize `minWgTiles[groupIdx] = 1` for each parallel
group. If any op requires a larger workgroup tile for a parallel group (e.g.,
ROCm matvec local-split-k), update with **lcm** to ensure the tile satisfies
all ops' divisibility requirements.

### 5. Assign workgroup tiles to root reduction op

Set `workgroup` tile sizes on the root reduction op only. For each parallel
group, use `minWgTiles[groupIdx]`. Translate to the root op's iteration dims
via `buildIterToGroupMap`.

```cpp
SmallVector<int> iterToGroup = analysis.buildIterToGroupMap(rootOp);
for (int iterDim = 0; iterDim < rootOp.getNumLoops(); iterDim++) {
    int gi = iterToGroup[iterDim];
    if (gi >= 0 && analysis.groups[gi].isParallelizable) {
        workgroupTileSizes[iterDim] = minWgTiles[gi];
    }
}
```

VectorDistribute infers workgroup tiling for other ops from the root.

---

## Problems Fixed

### Problem 1: Shared Parallel Dim Computation (populateConfigInfo, L406-423)

**Current:** Intersects iteration dim index sets. If op A has parallel dims
`{0, 1}` and op B has `{0, 1}`, intersection is `{0, 1}` — but index 1 in op
A (spatial) and index 1 in op B (channel/reduction) are different things.

**Fix:** Parallel groups come from `analysis.groups` where `isParallelizable ==
true`. Per-op translation via `buildIterToGroupMap`. No index-based intersection.

### Problem 2: Thread Distribution Target Dim (getVectorDistributeReductionConfig, L193-270)

**Current:** For parallel-only ops needing config, distributes on
`parallelDims.back()`. Only correct when iteration spaces happen to be aligned.

**Fix:** Use `buildIterToGroupMap(op)` to find the innermost iter dim mapping to
a reduction group. Distribute threads there.

### Problem 3: threadLoads Constraint (getThreadLoadsConstraint, L132-144)

**Current:** Global threadLoads is halved until it divides
`bounds[parallelDims.back()]` for every parallel-only op. This can use the
wrong dim (e.g., a batch dim of size 7) and force threadLoads to 1 globally.

**Fix:** Per-op threadLoads computed against the actual distribution dim (the
one in the reduction group). No global adjustment.

### Problem 4: Tile Sizes Passed by Raw Dim Index

**Current:** `sharedWgpTiles` maps iteration dim index → tile size. Applied to
every op using the same indices. Breaks when ops have different iteration spaces.

**Fix:** Tile sizes keyed by **group index**. Each op translates to its own
iteration dims via `buildIterToGroupMap`.

---

## What Stays the Same

- The overall structure: pre-check → compute globals → per-op config.
- Thread distribution formula: `partialReductionSize = gcd(wgSize * threadLoads, dimSize)`, then fit subgroup/thread basis.
- Dim expansion logic for vectorization (threadLoads > 1 with expand_shape).
- The `shouldAttachLoweringConfig` heuristics (broadcast, transpose, split users).
- ROCm matvec local-split-k logic (now updates `minWgTiles` via lcm instead of directly setting `sharedWgpTiles`).
- TranslationInfo with VectorDistribute pipeline, workgroup size, subgroup size.
