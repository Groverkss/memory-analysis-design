# Prototype V2: Overall Algorithm

## What v1 got right (keep)

- Tensor-centric parallelism via union-find dimension groups
- No dynamic shape special-casing (resolvedSize uniformly)
- Per-operand coalescing targets (subgroupSize * vectorWidth)
- Coalescing walk: innermost-to-outermost, reduction dims contribute for
  free, stop when next dim is non-contiguous
- HeuristicSeeds for tunability
- Observability via debug output

## What v1 got wrong (fix)

1. **Budget drops coalescing entirely** -- should progressively reduce
   vectorWidth instead, since benchmarks show all vector widths achieve
   the same load bandwidth
2. **No lane_basis/subgroup_basis production** -- analysis only, no config
   emission
3. **Single subgroup assumed** -- no multi-subgroup scaling
4. **Coalescing conflicts -> drop constraints** -- should use shared memory
   promotion instead
5. **Broadcast cost computed but unused** -- coarsening was removed
6. **Budget conflates parallel and sequential work** -- over-counts
   elementwise ops

## Key benchmark finding: vector width doesn't affect coalescing bandwidth

From the HIP coalescing benchmarks on RDNA4 (gfx1201) and CDNA3 (gfx942):

```
RDNA4 loads:   b16=611, b32=612, b64=612, b128=612 GB/s  (identical)
MI300X loads:  b16=3945, b32=3979, b64=4015, b128=3991 GB/s  (identical)
```

The hardware coalescer merges all requests within the same cache lines
regardless of per-thread vector width. Wider vectors reduce instruction
count but don't improve memory throughput.

For stores, wider vectors help with cache line contention:

```
RDNA4 stores:  b16=322, b32=532, b64=599, b128=600 GB/s  (gradual)
MI300X stores: b16=4739, b32=4846, b64=4873, b128=4918 GB/s  (mild)
```

**Implication:** Reducing vectorWidth from 8 to 4 (128-bit to 64-bit for
f16) is nearly free for loads and only mildly costly for stores. Much
better than dropping coalescing entirely.

## V2 Stages

```
Stage 1: Parallelism Analysis              -> dimension groups, resolved sizes
Stage 2: Coalescing + Tiling               -> per-operand min tiles, WG tiles, subgroups, per-op tiles
Stage 3: Budget Check                      -> progressively reduce vectorWidths if needed
Stage 4: Config Emission                   -> lowering_config per op
```

## Flow Diagram

input: tensor<1024x1024> // coalesce(vector_width)

```
analyzeParallelism()
        |
        v
  +---> computeCoalescingConstraints()     2a: per-operand min tiles
  |             |
  |     setWorkgroupTiles()                2b: WG tiles = parallel min tiles
  |             |
  |     scaleSubgroups()                   2c: subgroupsPerWG from occupancy
  |             |
  |     growTilesWithSubgroups()           2d: increase per-operand tiles
  |             |
  |     resolvePerOpTiles()               2e: max across operands per iter dim
  |             |
  |             v
  +---- checkBudget()                      3: if over budget or under-occupied,
                |                              halve worst vectorWidth, retry
                v
        emitPerOpConfigs()                 4: lowering_config per op
```

## Invariants

**After Stage 1:**
- Dimension groups classified as parallel or reduction.
- Each group has a resolvedSize (static or upper-bound).

**After Stage 2:**
- Each operand has min tile sizes per operand dim (from coalescing).
- Shared parallel dims have WG tiles = their min tile from coalescing.
- **No threads or subgroups distribute along shared parallel dims.**
  If you'd want distribution there, it should have been more WGs instead.
- Thread distribution is fully determined by the coalescing constraints:
  element_tile = vectorWidth on contiguous dim, threads fill reduction
  dims from innermost outward.
- Subgroups extend the tile from innermost outward (skip parallel dims).
- Per-op tiles are resolved from per-operand tiles (max across operands).

**After Stage 3:**
- Per-thread work is within budget for all ops.
- vectorWidths may have been reduced (never increased).
- numWGs >= some reasonable fraction of targetOccupancy.

---

## Stage 1: Parallelism Analysis

Unchanged from v1.

**Input:** All linalg ops in the dispatch.

TODO: Replace linalg-specific analysis with IndexingMapOpInterface.

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

## Stage 2: Coalescing + Tiling

**Input:** ParallelismAnalysis, HeuristicSeeds, per-operand vectorWidths.

### 2a. Per-operand coalescing constraints (min tile sizes)

For each global memory operand (dispatch inputs + stored outputs):

1. **Skip small operands.** If total size < significantOperandThreshold
   (seed, default 1KB), skip. Small operands fit in cache.

2. **Check if coalescing is possible.** Requires projected permutation
   indexing map and known contiguous innermost dim. If not, skip.

3. **Compute coalescingTarget:**

   ```
   coalescingTarget = subgroupSize * vectorWidth
   ```

   vectorWidth = maxLoadBits / elemBits (e.g., 8 for f16). May be smaller
   on retry from Stage 3.

4. **Walk operand dims innermost-to-outermost.** For each operand dim:
   - Map to its dimension group.
   - used = min(group.resolvedSize, remaining).
   - Record: this operand dim needs minTile = nextPowerOf2(used).
   - remaining -= used. If remaining <= 0: stop.
   - If next dim is non-contiguous (this dim's tile < full size): stop.

5. **Output per operand:**
   `{operand, vectorWidth, [(operandDim, minTile)...]}`.

The min tiles encode how the coalescingTarget is split across dims:
- Contiguous dim: vectorWidth (element_tile per thread).
- Next dims inward: thread distribution (subgroupSize fills here).
- Parallel dims get small min tiles (just vectorWidth if innermost).
- Reduction dims absorb the remaining target for free.

### 2b. Workgroup tile sizes

**WG tiles for shared parallel dims = their min tiles from coalescing.**

```
For each parallel group G:
  wgTile[G] = max of all operand min tiles that map to G
```

That's it. No further heuristics for WG tile sizes. Reduction groups
are not tiled across workgroups (wgTile = 0).

```
numWGs = product(resolvedSize / wgTile) for all parallel groups
```

### 2c. Subgroup scaling

Determine subgroupsPerWG from occupancy:

```
subgroupsPerWG = 1
while numWGs * subgroupsPerWG < targetOccupancy:
  subgroupsPerWG *= 2
  if subgroupsPerWG * subgroupSize > maxWorkgroupSize:
    subgroupsPerWG /= 2; break
```

### 2d. Grow per-operand tiles with subgroups

The coalescing walk (2a) gives min tiles assuming 1 subgroup (just
subgroupSize threads). With multiple subgroups, we can grow the tiles.

For each operand, distribute subgroups from innermost coalescing dim
outward, **skipping shared parallel dims** (those are WG-tiled, not
distributed within the WG):

```
remaining_subgroups = subgroupsPerWG
for dim in operand coalescing order (innermost first):
  group = findGroup(operand, dim)
  if group.isParallelizable:
    continue  // skip shared parallel dims
  // Grow this dim's tile by assigning subgroups
  dimHeadroom = group.resolvedSize / currentTile[dim]
  subgroupsHere = min(remaining_subgroups, dimHeadroom)
  currentTile[dim] *= subgroupsHere
  remaining_subgroups /= subgroupsHere
  if remaining_subgroups <= 1: break
```

### 2e. Resolve per-op tile sizes

Each operand has per-dim tile sizes. Map these to iteration space through
indexing maps, then resolve across operands:

```
For each compute op:
  For each iteration dim d:
    opTile[d] = 1
    For each operand of this op:
      operandDim = indexingMap.inverse(d)
      if operandDim exists and has a tile:
        opTile[d] = max(opTile[d], operandTile[operandDim])
```

For parallel dims: opTile = wgTile (from 2b).
For reduction dims: opTile = max across operands' grown tiles (from 2d).

These per-op tiles become the partial_reduction / serial tile sizes.

**Output:** Per-op tile sizes, per-operand coalescing configs (vectorWidth
+ order + min tiles), numWGs, subgroupsPerWG.

---

## Stage 3: Budget Check with Progressive VectorWidth Reduction

**Input:** Stage 2 output, HeuristicSeeds.

**Goal:** Ensure per-thread work is reasonable and occupancy is adequate.

### 3a. Compute per-op per-thread work

```
For each compute op:
  perThreadWork = product(opTiles) / (subgroupSize * subgroupsPerWG)
```

No reduction/elementwise distinction needed.

### 3b. Check triggers

Reduce vectorWidth when either trigger fires:

```
(a) perThreadWork > maxElementsPerThread     (over budget)
(b) numWGs < targetOccupancy                 (under-occupied)
```

### 3c. If triggered: reduce vectorWidth

1. **Score each active constraint** by how much halving its vectorWidth
   would improve the triggered metric.

2. **Halve the highest-scoring constraint's vectorWidth.**
   (e.g., f16: 8 -> 4 -> 2 -> 1)

3. **Go back to Stage 2** with updated vectorWidths. Recompute everything.

4. **Repeat** until both triggers are satisfied or all vectorWidths = 1.

5. **Last resort:** Drop constraint entirely (vectorWidth below 1). Rare.

### 3d. Coalescing conflicts

Two operands conflict when they need different dims coalesced. Both
keep their coalescing specs. VectorDistribute resolves by inserting
shared memory copies for the operand that doesn't match the chosen
iteration-space layout.

**Output:** Final per-operand vectorWidths, final per-op tile sizes.

---

## Stage 4: Config Emission

**Input:** All previous results.

**Goal:** Produce a `lowering_config` for each compute op.

### 4a. Which ops get configs

- Reduction ops: always.
- Elementwise ops with broadcast inputs or non-trivial output maps: yes.
- Simple elementwise fuseable into a reduction's tile loop: no (fused).

### 4b. For each op with a config

- **workgroup**: parallel dim tiles from 2b. Reduction dims = 0.

- **partial_reduction** (reduction ops): reduction tiles from 2e.
  Parallel dims = 0.

- **serial** (elementwise ops): reduction dim sizes. Parallel dims = 0.

- **thread** (element_tile): vectorWidth on innermost coalescing dim.
  1 on other distributed dims. 0 elsewhere.

- **lane_basis**: derived from the per-operand coalescing config of
  the largest operand, projected to iteration space. Threads fill
  from innermost coalescing dim outward, skipping parallel dims.

- **subgroup_basis**: same derivation, for the subgroup level.

- **operand_configs**: per-operand coalescing specs (vectorWidth, order,
  min tiles). ConfigureTensorLayouts uses these to set per-operand
  NestedLayoutAttrs. VectorDistribute inserts shared memory copies
  where layouts disagree.

TODO: Define `operand_config` attribute format in IREEGPUAttrDefs.td.

**Output:** `lowering_config` attribute attached to each op.

---

## Worked Examples

### Innermost Reduction: 1024x4096 f16

```
Groups: P0=1024 (parallel), R0=4096 (reduction)

2a: input <P0 x R0>: contiguous=R0. target=512.
    R0 (reduction): 4096 >= 512. Done.
    min tiles: [P0=1, R0=512]. vectorWidth=8 on R0.

2b: wgTile[P0] = 1 (no parallel constraint). numWGs = 1024.

2c: 1024 WGs, target=1216. subgroupsPerWG = 2.

2d: Grow with 2 subgroups (skip P0, grow R0):
    input: R0 = 512 * 2 = 1024.

2e: reduce op tiles: [P0=1, R0=1024].
    perThreadWork = 1 * 1024 / 128 = 8. Well within budget.

Config: workgroup=[1, 0], partial_reduction=[0, 1024], thread=[0, 8]
Thread distribution: 64 threads on R0 per subgroup, 2 subgroups on R0.
```

### Outermost Reduction: 4096x1024 f16

```
Groups: P0=1024 (parallel), R0=4096 (reduction)

2a: input <R0 x P0>: contiguous=P0. target=512.
    P0 (parallel): min=8 (vectorWidth). remaining=512/8=64.
    R0 (reduction): 4096 >= 64. Done.
    min tiles: [R0=64, P0=8].

2b: wgTile[P0] = 8. numWGs = 1024/8 = 128.

2c: 128 WGs, target=1216. subgroupsPerWG = 16 (128*16=2048).
    But 16*64=1024=maxWG. subgroupsPerWG = 16.

2d: Grow with 16 subgroups (skip P0, grow R0):
    input: R0 = 64 * 16 = 1024.

2e: reduce op tiles: [R0=1024, P0=8].
    perThreadWork = 1024 * 8 / 1024 = 8. Within budget.

Config: workgroup=[0, 8], partial_reduction=[1024, 0], thread=[1, 8]
128 WGs, 1024 threads each. 64 threads on R0 per subgroup, vec8 on P0.
```

### Matvec: 8x1024 @ 1024x1024 f16

```
Groups: P0=M=8, P1=N=1024, R0=K=1024

2a: lhs <P0 x R0>: contiguous=R0. target=512.
    R0 (reduction): 1024 >= 512. Done.
    min tiles: [P0=1, R0=512].

    rhs <R0 x P1>: contiguous=P1. target=512.
    P1 (parallel): min=8 (vectorWidth). remaining=64.
    R0 (reduction): 1024 >= 64. Done.
    min tiles: [R0=64, P1=8].

2b: wgTile[P0]=1, wgTile[P1]=8. numWGs = 8 * 128 = 1024.

2c: 1024 WGs, target=1216. subgroupsPerWG = 2.

2d: Grow with 2 subgroups (skip P0, P1; grow R0):
    lhs: R0 = 512 * 2 = 1024.
    rhs: R0 = 64 * 2 = 128.

2e: contraction op tiles (M, N, K):
    From lhs (M,K): M=1, K=1024.
    From rhs (K,N): K=128, N=8.
    Resolve: M=1, N=8, K=max(1024,128)=1024.
    perThreadWork = 1 * 8 * 1024 / 128 = 64. Within budget.

Config: workgroup=[1, 8, 0], partial_reduction=[0, 0, 1024], thread=[0, 8, 1]
1024 WGs, 128 threads each.
lhs layout: vec8 on K, 64 threads on K. Coalesced along K.
rhs layout: vec8 on N, 64 threads on K. Stride=8*2=16B. Cache-coalesced.
Layouts conflict -> VectorDistribute inserts shmem for one operand.
```

### Elementwise: 1024x4096 f16

```
Groups: P0=1024, P1=4096 (both parallel)

2a: input <P0 x P1>: contiguous=P1. target=512.
    P1 (parallel): min=512 (no reduction dims to fill the rest).
    min tiles: [P0=1, P1=512].

2b: wgTile[P0]=1, wgTile[P1]=512. numWGs = 1024 * 8 = 8192.

2c: 8192 >> 1216. subgroupsPerWG = 1.

2d: No reduction dims to grow. Tiles stay at [1, 512].

2e: elementwise op tiles: [P0=1, P1=512].
    perThreadWork = 1 * 512 / 64 = 8. Within budget.

Config: workgroup=[1, 512], serial=[0, 0], thread=[0, 8]
8192 WGs, 64 threads each. Threads along P1 (coalesced). Standard.
```

Note: for elementwise, there are no reduction dims, so the full
coalescingTarget must come from parallel dims. The parallel dim gets
the full target (512, not weakened to vectorWidth). The weakening only
happens when reduction dims can absorb the remaining target.
