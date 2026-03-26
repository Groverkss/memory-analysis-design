# Prototype V2: Overall Algorithm

## What v1 got right (keep)

- Tensor-centric parallelism via union-find dimension groups
- No dynamic shape special-casing (resolvedSize uniformly)
- Per-operand coalescing targets (subgroupSize * 128/elemBits)
- Coalescing walk: innermost-to-outermost, reduction dims contribute for
  free, stop when next dim is non-contiguous
- HeuristicSeeds for tunability
- Observability via debug output

## What v1 got wrong (fix)

1. **Budget conflates parallel and sequential work** -- drops coalescing on
   elementwise transpose when it shouldn't
2. **No lane_basis/subgroup_basis production** -- analysis only, no config
   emission
3. **Single subgroup assumed** -- no multi-subgroup scaling
4. **Coalescing conflicts -> drop constraints** -- should use shared memory
   promotion instead
5. **Broadcast cost computed but unused** -- coarsening was removed

## V2 Stages

```
Stage 1: Parallelism Analysis        -> dimension groups, resolved sizes
Stage 2: Workgroup Tiling            -> per-group WG tiles, WG count
Stage 3: Per-Thread Work Budgeting   -> validate/adjust tiles, flag promotions
Stage 4: Thread & Subgroup Layout    -> lane_basis, subgroup_basis, promote_operands
Stage 5: Per-Op Config Emission      -> lowering_config per op
```

## Flow Diagram

```
analyzeParallelism()
        |
        v
computeWorkgroupTiling()  <-- coalescing constraints + broadcast cost
        |
        v
checkPerThreadBudget()    <-- detect conflicts, flag promotions
        |
        v
buildThreadLayout()       <-- lane_basis, subgroup_basis
        |
        v
emitPerOpConfigs()        <-- lowering_config per op
```

No outer retry loop. Conflicts are resolved by promotion in Stage 3, not by
iteratively dropping constraints.

---

## Stage 1: Parallelism Analysis

Unchanged from v1. This is solid.

**Input:** All linalg ops in the dispatch.

**Steps:**

1. Gather compute ops (backward slice from stores, exclude fills).

2. Build tensor infos (per-tensor access patterns, indexing maps).

3. Build dimension groups via union-find (link tensor dims that map to the
   same iteration dim within any op). A group is parallel only if ALL
   members are parallel in every op.

4. Resolve dynamic dims (IntegerRangeAnalysis for upper bounds, default 1M
   for truly unknown).

**Output:** `ParallelismAnalysis` -- computeOps, tensorInfos, groups (each
with isParallelizable, resolvedSize).

---

## Stage 2: Workgroup Tiling

**Input:** ParallelismAnalysis, HeuristicSeeds.

**Goal:** Decide how to tile parallel groups across workgroups.

### 2a. Per-operand coalescing constraints

For each global memory operand (dispatch inputs + stored outputs):

1. **Check if coalescing is possible.** Coalescing requires the operand's
   indexing map to be a projected permutation and the memory layout to have
   a known contiguous innermost dimension. If not, skip this operand — no
   coalescing constraint generated.

2. **Compute coalescingTarget** = subgroupSize * (128 / elemBits).

3. **Walk operand dims innermost-to-outermost.** For each operand dim:
   - Map the operand dim to its dimension group.
   - The "used" amount from this dim = min(group.resolvedSize, remaining).
   - Subtract the used amount from remaining.
   - Record: this operand dim needs a minimum tile of
     nextPowerOf2(used amount).
   - If remaining <= 0: stop, coalescing satisfied.
   - If the next dim's stride would make it non-contiguous with this dim
     (i.e., this dim's tile < this dim's full size, creating a gap), stop.
     Coalescing from outer dims is not possible.

4. **Output per constraint:** `{operand, [(operandDim, minTile)...]}`.
   These constraints are on *operand dimensions*, not on iteration space
   dimensions or dimension groups. The mapping to groups/iteration space
   happens later.

Note: parallel and reduction dims are not treated differently here. A
constraint just says "this operand dimension needs tile >= X". What that
*costs* is determined later:
- A parallel dim constraint costs parallelism (larger WG tile).
- A reduction dim constraint costs a partial_reduction tile minimum.

### 2b. Map constraints to WG tiles

For each parallel dimension group G:

- Collect all coalescing constraints that touch G (by mapping each
  constraint's operand dims to their dimension groups).
- wgTile[G] = max of all constraint minTiles that map to G.

This is where the max across operands happens — not in 2a.

For reduction groups: constraints are noted but don't affect WG tiling.
They flow to Stage 3 as partial_reduction tile minimums.

### 2c. Dimension interactions (broadcast cost)

For each parallel group G:

- broadcastCost(G) = total elements of operands that do NOT have a dim
  in G.

- This informs coarsening: tiling G aggressively wastes bandwidth on
  broadcast operands.

### 2d. WG count, coarsening, and subgroup scaling

**Compute initial WG count:**

- numWGs = product(resolvedSize / wgTile) for all parallel groups.

**Coarsening (if too many WGs):**

- Coarsening packs multiple "rows" of work into a single WG along
  parallel groups.
- Cap: no parallel group's wgTile may exceed
  maxParallelRowsPerWG * coalescing minimum for that group (seed,
  default 8). This prevents a single WG from processing a huge number
  of independent rows.
- Preference order for which group to coarsen:
  1. Groups NOT critical for coalescing (no constraint touches them).
  2. Among those, prefer groups with high broadcast cost.
- Double the tile of the best group, repeat until numWGs <=
  coarsenThreshold * targetOccupancy or all groups are at their cap.

**Subgroup scaling (if too few WGs):**

- subgroupsPerWG = 1.
- While numWGs * subgroupsPerWG < targetOccupancy:
  subgroupsPerWG *= 2.
- Cap at maxWorkgroupSize / subgroupSize.

**Output:** `WorkgroupTiling` -- groupTiles, numWGs, subgroupsPerWG,
perOperandConstraints, broadcastCosts.

---

## Stage 3: Per-Thread Work Budgeting

**Input:** ParallelismAnalysis, WorkgroupTiling (including per-operand
constraints), HeuristicSeeds.

**Goal:** Validate that per-thread work is reasonable. If not, resolve by
adjusting tiles or flagging shared memory promotion. This replaces v1's
broken per-WG budget check.

### Key v1 fix: separate elementwise from reduction work

- For a **reduction** op: per-thread sequential work =
  product(reduction tiles). This bounds how many accumulation iterations
  each thread does.

- For an **elementwise** op: work is distributed across all threads.
  Per-thread work = product(all tiles) / numThreads. Much more permissive.

### 3a. Compute reduction tiles from constraints

For each reduction dimension group R:

- Collect coalescing constraints from 2a that map to R.
- reductionTile[R] = max of those constraint minTiles.
- This is the minimum partial_reduction tile needed for coalescing on
  operands that have R as an innermost dim.

### 3b. Compute per-op per-thread work

For each compute op:

- Compute tiles: parallel tiles from WG tiling (Stage 2b), reduction
  tiles from 3a.

- Classify op: has reduction dims -> reduction budget; all parallel ->
  elementwise budget.

- Reduction budget: `reductionWork = product(reduction_tiles)`.
  Target: <= maxReductionIterationsPerThread (seed, e.g., 1024).

- Elementwise budget:
  `elemWork = product(all_tiles) / (subgroupsPerWG * subgroupSize)`.
  Target: <= maxElementsPerThread (seed, e.g., 4096).

### 3c. Detect coalescing conflicts

- Two operands conflict when they need different parallel groups coalesced
  (e.g., input wants P1 innermost, output wants P0 innermost).

- v1 dropped constraints. v2: flag the non-dominant operand for shared
  memory promotion (promote_operands).

- Only drop constraints when promotion isn't viable (shared memory budget
  exceeded).

### 3d. If over budget

- For reduction ops over budget: reduce the largest reduction tile (fewer
  serial iterations per thread).

- For elementwise ops over budget: this is rare since the budget is
  permissive; if it happens, coarsen WG tiles.

- For coalescing conflicts: resolve via promotion, not by dropping.

**Output:** `BudgetResult` -- adjusted reduction tiles per op,
promote_operands set, any dropped constraints.

---

## Stage 4: Thread & Subgroup Layout

**Input:** ParallelismAnalysis, WorkgroupTiling, BudgetResult,
HeuristicSeeds.

**Goal:** Produce lane_basis and subgroup_basis that achieve coalesced
access for the dominant operand, and mark operands that need promotion.

### 4a. Determine dominant operand

The dominant operand is the largest per-WG global memory operand for the
dispatch.

### 4b. Build lane_basis

From the dominant operand's memory layout:

- Walk dominant operand dims innermost-to-outermost.
- Map each operand dim -> iteration dim via indexing map.
- Assign threads: innermost dim gets vectorWidth elements per thread, fill
  subgroupSize threads from innermost outward.
- Reduction dims in the thread distribution -> cooperative shuffle
  reduction.
- Parallel dims in the thread distribution -> independent work.

### 4c. Build subgroup_basis

- Subgroups extend the thread distribution along the same dimensions.
- If multiple subgroups: place them along the coalescing dimension
  (increases per-WG bandwidth without changing the coalescing pattern).

### 4d. Mark promote_operands

- Operands whose coalescing layout conflicts with the lane_basis get
  promote_operands annotation.
- ConfigureTensorLayouts will insert shared memory conversions for these.

**Output:** `ThreadLayout` -- lane_basis, subgroup_basis,
promote_operands, vectorWidth per op.

---

## Stage 5: Per-Op Config Emission

**Input:** All previous results.

**Goal:** Produce a `lowering_config` for each compute op that needs one.

### 5a. Which ops get configs

- Reduction ops: always get a config.
- Elementwise ops with broadcast inputs or non-trivial output maps: get
  a config.
- Simple elementwise fuseable into a reduction's tile loop: no config
  (fused).

### 5b. For each op with a config

- **workgroup**: parallel dim tiles from Stage 2 groupTiles, mapped
  through op's indexing map. Reduction dims = 0.

- **partial_reduction** (reduction ops): reduction tiles from Stage 3a,
  mapped through op's indexing map. Parallel dims = 0.

- **serial** (elementwise ops): reduction dim sizes (these are serial
  loops). Parallel dims = 0.

- **thread**: vectorWidth on the innermost coalescing dim. 1 on other
  distributed dims. 0 elsewhere.

- **lane_basis**: from Stage 4, projected to this op's iteration space.

- **subgroup_basis**: from Stage 4, projected to this op's iteration
  space.

- **promote_operands**: from Stage 3/4, indices of operands needing
  shared memory.

**Output:** `lowering_config` attribute attached to each op.
