# Design v2: Memory-Bound Operation Configuration

## Core Principle (unchanged)

**Thread distribution is driven by memory coalescing, not by iterator types.**

## Key Corrections from v1

1. **Coalescing is per-operand, not per-iteration-space-dim.** We don't pick
   a single "dominant operand." Each operand has its own coalescing needs.
   If operands disagree on the ideal thread layout, VectorDistribute
   automatically inserts shared memory conversions. We don't need to resolve
   conflicts ourselves.

2. **Coalescing spans multiple dimensions.** A thread's coalesced access
   region covers the innermost contiguous dimensions of the operand, not
   just one dim. Threads fill from the innermost dim outward.

3. **Always use max vector width.** `vectorWidth = maxLoadBits / bitWidth`.
   Period. Masking handles non-divisible sizes. No adjustment loop.

4. **Subgroup scaling: start small, grow if needed.** Default to 1 subgroup
   per workgroup. Only add more subgroups if there isn't enough parallel
   work (workgroups) to fill the GPU.

## The Coalescing Model

For an operand `tensor<D0 x D1 x ... x Dn>` in row-major layout:
- Dn is the innermost (contiguous) dimension
- D(n-1) is the next contiguous dimension (stride = Dn)
- etc.

A **coalesced thread distribution** for this operand distributes threads
starting from Dn outward:
```
threads_on_Dn    = min(ceil(Dn / vectorWidth), remaining_threads)
threads_on_D(n-1) = min(D(n-1), remaining_threads / threads_on_Dn)
threads_on_D(n-2) = ...
```

Each operand's indexing map determines which iteration space dimensions
these correspond to. The coalescing layout is specified in terms of the
operand, then projected to the iteration space.

## Algorithm

### Step 1: Compute vector width

```
vectorWidth = maxLoadBits / elementBitWidth
```

No adjustment for divisibility. Masking handles the rest.

### Step 2: Determine coalesced thread layout per operand

For each operand with an indexing map that is a projected permutation:

a. Find the operand's contiguous dimensions (innermost first):
   `operand_dims = [rank-1, rank-2, ...]`

b. Map each through the indexing map to iteration space dims:
   ```
   coalescing_iter_dims = []
   for operand_dim in operand_dims (innermost first):
     iter_dim = indexing_map.getDimPosition(operand_dim)
     coalescing_iter_dims.append(iter_dim)
   ```

c. Distribute threads along these dims (innermost gets stride 1):
   ```
   remaining = subgroupSize
   for iter_dim in coalescing_iter_dims:
     dim_size = bounds[iter_dim]
     threads_here = min(ceil(dim_size / vectorWidth_for_this_dim), remaining)
     // vectorWidth only applies to the innermost dim; others get 1-element threads
     remaining /= threads_here
     if remaining == 1: break
   ```

### Step 3: Choose a thread layout

Since each operand may want different thread distributions, we need to pick
one for the iteration space (the lane_basis). But we DON'T need to agonize
over conflicts — VectorDistribute handles mismatches with shared memory.

Heuristic for picking: **use the coalesced layout of the largest input operand.**
Inputs dominate memory traffic for reductions (read in a loop). The output is
written once and is typically smaller.

For element-wise (no reduction): all operands have similar traffic, pick
the one with the most complex indexing (most likely to cause strided access
if not prioritized).

Alternatively, we could specify coalescing per-operand in the config and let
ConfigureTensorLayouts handle it. But for now, picking one layout for the
iteration space and relying on shared memory for conflicts is simpler.

Comment: I do think that specifying coalescing per-operand is a better way of
doing things.

### Step 4: Build lane_basis

The lane_basis encodes the chosen thread distribution. The coalescing
dimensions go at the END of the counts array (so they get the lowest strides),
with the innermost dim last (stride = 1).

```
counts  = [..., threads_on_next_dim, threads_on_innermost_dim]
mapping = [..., next_iter_dim,       innermost_iter_dim]
```

Example: `tensor<M x N x K>` with K contiguous, threads spanning K and N:
```
lane_basis = [[threads_on_N, threads_on_K], [iter_dim_N, iter_dim_K]]
```
Strides: threads_on_K for N, 1 for K. So thread IDs map linearly to K
first, then N — matching memory order.

### Step 5: Subgroup distribution

Start with 1 subgroup per workgroup. Then check if we have enough
workgroups to fill the GPU:

```
numWorkgroups = product(parallel_dim_sizes) / product(workgroup_tiles)
target = numWGPs * numSIMDs  // or some fraction

subgroups_per_wg = 1
while numWorkgroups * subgroups_per_wg < target:
    subgroups_per_wg *= 2
    if subgroups_per_wg * subgroupSize > maxWorkgroupSize:
        subgroups_per_wg /= 2
        break
```

Subgroups are placed along the coalescing dimension (same dims as threads).
This increases the per-workgroup tile without changing the coalescing pattern.

### Step 6: Workgroup tile sizes

For parallel dims: tile to create enough workgroups.
For reduction dims: 0 (entire reduction within the workgroup).

```
workgroup[parallel_dim] = some_tile_size  // e.g., 1 for the "batch" dim of layernorm
workgroup[reduction_dim] = 0
```

The parallel tile sizes should balance:
- Enough WGs to fill the GPU
- Each WG has enough work to amortize launch overhead

### Step 7: Partial reduction tile sizes

This is the chunk size for the reduction loop body. It determines the vector
size after vectorization.

```
// Total elements processed by the workgroup per chunk:
chunk_size = subgroups_per_wg * subgroupSize * vectorWidth

partial_reduction[reduction_dim] = chunk_size  (on the distributed reduction dim)
partial_reduction[other_reduction_dims] = some_serial_tile  (or full size)
partial_reduction[parallel_dims] = 0
```

For reduction dims that are NOT distributed across threads (serial loops),
the partial_reduction tile controls how many serial iterations happen
before the next vector load. These become the batch dimension in
NestedLayout.

### Step 8: Thread tile sizes

```
thread[innermost_coalescing_dim] = vectorWidth
thread[other_dims] = 1 or 0
```

## Worked Examples (revised)

### Example 1: Innermost Reduction — tensor<1024 x 4096 x f16> → tensor<1024>

Input indexing map: (d0,d1) → (d0,d1). Contiguous operand dim = 1 → iter dim d1.

```
vectorWidth = 128/16 = 8
threads on d1 = min(ceil(4096/8), 64) = 64   // fills subgroup
lane_basis = [[64], [1]]  // 64 threads on d1, stride 1

subgroups: parallelSize = 1024 rows → 1024 WGs if wg_tile=[1,0]
  → plenty of WGs, start with 1 subgroup
  → actually 1024 might not be enough if numWGPs*numSIMDs > 1024,
    so maybe 2 or 4 subgroups
  subgroup_basis = [[4], [1]]  (4 subgroups on d1)

workgroup = [1, 0]
partial_reduction = [0, 4*64*8] = [0, 2048]
thread = [0, 8]
```

Same as current config ✓.

### Example 2: Outermost Reduction — tensor<4096 x 1024 x f16> → tensor<1024>

Input indexing map: (d0,d1) → (d0,d1). Contiguous operand dim = 1 → iter dim d1.
d1 is PARALLEL.

```
vectorWidth = 8
threads on d1 = min(ceil(1024/8), 64) = 64
remaining = 1
lane_basis = [[64], [1]]  // threads on d1 (parallel!)

subgroups: workgroup tile for d1 = 64*8 = 512 → 1024/512 = 2 WGs
  → probably not enough, increase subgroups
  → 2 subgroups: each WG does 2*512 = 1024 → 1 WG total... worse
  → actually subgroups should go along d1 too
  Wait, if subgroups go along d1, that increases the WG tile, REDUCING
  the number of WGs. That's backwards.

  Actually for outermost reduction, the parallel dim is d1 and we want
  MORE workgroups along d1. Subgroups don't help here. The right move is
  to use a smaller workgroup tile along d1 to get more WGs:

  workgroup_tile_d1 = 64 * 8 = 512 → 2 WGs
  Or: workgroup_tile_d1 = 64 * 8 / 4 = 128 → 8 WGs (but this means only 16 threads?)

  Actually the workgroup tile for d1 is determined by the thread + subgroup
  distribution. With 1 subgroup, 64 threads, 8 vec width: 512 per WG.

  For 1024 columns, that's 2 WGs. If GPU has 512 WGPs, that's way too few.

  Options:
  a) Just accept 2 WGs (if reduction is large enough per WG, memory
     bandwidth is the bottleneck anyway)
  b) Use smaller workgroup (fewer threads) → more WGs
  c) Tile d1 to 128: each WG does tensor<4096 x 128>
     128 / 8 = 16 threads, subgroupSize = 64 → 16 threads underutilize
     But remaining 48 threads could go on d0 (reduction):
     threads_on_d0 = min(4096, 64/16) = 4
     → lane_basis = [[4, 16], [0, 1]]
     → 4 threads on d0, 16 threads on d1, total = 64
     → d0 is reduction: threads cooperate via shuffles
     → d1 is parallel: independent columns
     → 1024/128 = 8 WGs

  But wait, d0 threads access strided memory (different rows). Each thread
  reads 8 contiguous elements along d1 (coalesced), at different rows.
  The d0 distribution doesn't affect coalescing of individual loads.
  The coalescing is still fine because within each thread's load,
  it reads 8 contiguous elements along d1.

  Between threads at the same d0 position but adjacent d1 positions:
  thread 0 reads cols [0:8], thread 1 reads cols [8:16] → contiguous ✓

  Between threads at adjacent d0 positions but same d1 position:
  thread 0 reads row 0 cols [0:8], thread 16 reads row 1 cols [0:8]
  → these are stride-1024 apart in memory → not contiguous, but that's
  the d0 dimension which is NOT the coalescing dim.

  Within a warp/subgroup, the first 16 threads (d0=0) read cols 0-127
  contiguously → coalesced ✓. The next 4 threads (d0=1,2,3 at d1=0)
  Wait, the thread ordering depends on strides.

  lane_basis = [[4, 16], [0, 1]]
  strides = [16, 1]  // d1 gets stride 1, d0 gets stride 16

  thread 0: d0=0, d1=0  → row 0, cols [0:8]
  thread 1: d0=0, d1=1  → row 0, cols [8:16]
  ...
  thread 15: d0=0, d1=15 → row 0, cols [120:128]
  thread 16: d0=1, d1=0  → row 1, cols [0:8]
  thread 17: d0=1, d1=1  → row 1, cols [8:16]
  ...

  Threads 0-15 all read from row 0, contiguous cols → coalesced ✓
  Threads 16-31 all read from row 1, contiguous cols → coalesced ✓

  Within a 64-thread subgroup:
  - Threads 0-15: row 0 (coalesced)
  - Threads 16-31: row 1 (coalesced)
  - Threads 32-47: row 2 (coalesced)
  - Threads 48-63: row 3 (coalesced)

  GPU coalescing works per-warp (32 threads on NVIDIA, 64 on AMD).
  On AMD (64): threads 0-15 and 16-31 issue loads to different rows,
  but within each group of 16, addresses are contiguous. The hardware
  can coalesce within each group.

  This is a reasonable tradeoff. Not perfectly sequential but each
  "chunk" is coalesced.
```

So for outermost reduction, the approach is:
- Pick a workgroup tile for d1 that creates enough WGs
- Distribute threads: some on d1 (coalescing), rest on d0 (if needed)
- lane_basis encodes this multi-dim distribution
- d0 threads cooperate via shuffle for partial reduction within the chunk

```
workgroup = [0, 128]     // 8 WGs
partial_reduction = [64, 0]  // 64 rows per chunk (4 threads × 16 batch)
thread = [1, 8]          // 8 contiguous f16 along d1
lane_basis = [[4, 16], [0, 1]]
subgroup_basis = [[1, 1], [0, 1]]
```

NestedLayout for vector<64 x 128 x f16>:
```
subgroup_tile = [1, 1]
batch_tile    = [16, 1]   ← 16 rows serial per thread (64 / 4 threads on d0)
thread_tile   = [4, 16]
element_tile  = [1, 8]
```
Check: 1 * 16 * 1 * 4 * 1 = 64 ✓, 1 * 1 * 1 * 16 * 8 = 128 ✓

Reduction (dim 0): threadTile[0]=4 → shuffle reduce across 4 threads.
Each thread first reduces its 16 batch rows locally, then shuffle-reduces
with the other 3 threads on d0.

Memory: each thread reads 8 contiguous f16 per load (128-bit), threads at
adjacent d1 positions read adjacent memory → coalesced within groups ✓.

### Example 3: Batch Norm — tensor<64 x 256 x 56 x 56 x f16> → tensor<256>

Input contiguous dim = d3 (W=56).

```
vectorWidth = 8
threads on d3 = ceil(56/8) = 7

Remaining = 64/7 ≈ 9... doesn't divide cleanly.
```

With masking (v2 approach): just use vectorWidth=8, and let masking handle
the fact that 56 isn't divisible by 8. We'd have ceil(56/8) = 7 "thread
positions" along d3, but 7 doesn't divide 64.

Alternative: span into d2 (H=56):
```
threads on d3 = 7 (with masking for 56 % 8 ≠ 0... wait, 56 / 8 = 7 exactly)
Actually 7 * 8 = 56. So 7 threads, each doing 8 elements. No masking needed.

remaining = 64 / 7 ≈ 9.1 → can't divide 64 evenly by 7.
```

This is the core problem with 56: it factors as 2^3 × 7. The subgroup size
(64 = 2^6) isn't divisible by 7.

**Practical resolution**: the masking approach lets us use 8 threads on d3
(8 * 8 = 64 > 56, last thread is partially masked), with remaining = 64/8 = 8
threads on d2. Total: 64 threads.

```
threads on d3 = 8 (padded; last thread handles 0 elements, masked)
threads on d2 = 8
lane_basis = [[8, 8], [2, 3]]
```

Each thread reads 8 contiguous elements along d3 (W). 8 threads cover
8*8 = 64 positions along W, but only 56 are valid → masking.
8 threads along d2 (H=56): each handles ceil(56/8) = 7 rows.

The remaining reduction dims (d0=N=64) are serial loops.

```
workgroup = [0, 1, 0, 0]       // 1 C-channel per WG → 256 WGs
partial_reduction = [1, 0, 7, 64]  // or whatever tiling
thread = [0, 0, 1, 8]
lane_basis = [[8, 8], [2, 3]]
subgroup_basis = [[1, 1], [2, 3]]
```

This is cleaner than v1's struggle with 56. Masking makes it tractable.

## Summary of Changes from v1

| Aspect | v1 | v2 |
|--------|----|----|
| Coalescing target | Single dominant operand, single iter dim | Per-operand, multi-dimensional |
| Vector width | Adjusted for divisibility | Always max; masking handles rest |
| Subgroup scaling | Heuristic threshold | Start at 1, double until enough WGs |
| Split-k | Mentioned as option | Out of scope |
| Non-power-of-2 dims | Struggled (batch norm) | Masking makes it tractable |
| Layout conflicts | Must resolve | VectorDistribute handles via shared memory |

## What the Config Looks Like

For the outermost reduction example:
```mlir
#iree_gpu.lowering_config<{
  workgroup = [0, 128],
  partial_reduction = [64, 0],
  thread = [1, 8],
  lane_basis = [[4, 16], [0, 1]],
  subgroup_basis = [[1, 1], [0, 1]]
}>
```

This config says:
- Workgroup tiles 128 columns, processes all rows
- Reduction loop: 64 rows per chunk
- Each thread reads 8 contiguous f16 along d1
- 4 threads on d0 (reduction, stride 16), 16 threads on d1 (parallel, stride 1)
- 1 subgroup

The pipeline interprets this identically to today — no changes needed in
ConfigureTensorLayouts or VectorDistribute.

## Code Changes

**ReductionConfigUtils.cpp**: Rewrite the core logic:

```cpp
LogicalResult setMemoryBoundConfig(TargetAttr target,
                                    FunctionOpInterface entryPoint,
                                    linalg::LinalgOp op) {
  // 1. Vector width: always max
  int64_t bitWidth = getBitWidth(op);
  int64_t vectorWidth = maxLoadBits / bitWidth;

  // 2. Coalesced thread distribution
  //    Walk operand dims from innermost out, map to iteration space,
  //    fill threads.
  Basis laneBasis = buildCoalescedBasis(op, vectorWidth, subgroupSize);

  // 3. Subgroup scaling
  int64_t subgroupsPerWg = 1;
  int64_t numWGs = estimateWorkgroupCount(op, laneBasis, subgroupsPerWg);
  while (numWGs < targetOccupancy && canDoubleSubgroups(...)) {
    subgroupsPerWg *= 2;
    numWGs = estimateWorkgroupCount(op, laneBasis, subgroupsPerWg);
  }

  // 4. Set all config fields
  ...
}
```

**No changes** to ConfigureTensorLayouts or distribution patterns.

## Open Questions (revised)

1. **Which operand to base the iteration-space layout on?** We said
   VectorDistribute handles conflicts, but we still need to pick ONE
   layout for the iteration space. Largest input is a reasonable default.

2. **Should we add per-operand coalescing hints to the config?** This would
   let ConfigureTensorLayouts set different layouts per operand instead of
   projecting a single iteration-space layout. Not needed for v1 but could
   improve things later.

3. **Interaction with fused dispatches**: Multiple ops sharing workgroup
   tiles need compatible thread distributions. Same constraint as today.

4. **Rename**: `setReductionConfig` → `setMemoryBoundConfig`.
