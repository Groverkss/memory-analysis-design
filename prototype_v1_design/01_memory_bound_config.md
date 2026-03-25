# Design: Memory-Bound Operation Configuration

## Problem Statement

The current `setReductionConfig` always distributes threads along the last
reduction dimension. This works when the reduction dim happens to be the
contiguous memory dimension, but produces strided (uncoalesced) access when
it isn't. We need a config strategy that:

1. Distributes threads based on **operand memory layout**, not iterator types
2. Handles innermost-dim reduction, outermost-dim reduction, and mixed cases
3. Is extensible to scan, sort, topk, argmax

## Core Principle

**Thread distribution is driven by memory coalescing, not by whether a
dimension is parallel or reduction.**

For coalesced global memory access, consecutive threads must access
consecutive memory addresses. This means threads should be distributed
along the **contiguous (innermost) dimension of the dominant operand**.

Whether that dimension is a reduction dim or a parallel dim determines
the *reduction strategy*, not the *thread layout*.

- If threads are along a **reduction** dim → cooperative reduction (shuffles)
- If threads are along a **parallel** dim → independent work, reduction is
  serial per thread (loop accumulation)

Both strategies are already supported by the existing VectorDistribute pipeline.

## Algorithm

### Step 1: Identify the coalescing target

For each operand of the linalg op:
- Find its innermost (contiguous) tensor dimension (last dim for row-major)
- Map that through the operand's indexing map to find the corresponding
  iteration space dimension

Pick the **dominant operand** — the one with the most memory traffic:
- For reductions: the input tensor (read in every iteration of the reduction
  loop; output is written once)
- For element-wise fused ops: the largest operand
- Tie-break: prefer the operand whose coalescing dim has the largest size

Comment: I don't think we need to pick a single dominant operand. There can be
multiple operands that need coalescing. If the layouts don't agree, it's ok,
vector distribute will automatically insert shared memory for this kind of
stuff. It's not something you need to worry. I also don't think we need to 
realistically consider the iteration space dimension. The coalescing can span
multiple dimensions.

Result: `coalescing_iter_dim` — the iteration space dimension that, when
threads are distributed along it, gives coalesced access for the dominant
operand.

### Step 2: Determine vector width

```
vectorWidth = maxLoadBits / elementBitWidth
```
(e.g., 128 / 16 = 8 for f16)

Adjust down until `coalescing_dim_size % vectorWidth == 0`. Also ensure
all other ops in the dispatch have compatible constraints.

Comment: We don't need to do this. We should always pick coalescing vector width.
Masking will automatically handle if the size isn't divisible. The old code didn't 
understand this well.

### Step 3: Thread distribution (lane_basis)

The coalescing dim gets the **lowest stride** (stride = 1) in the thread
layout. This is achieved by placing it LAST in the basis counts array.

```
available_threads = subgroupSize
threads_on_coalescing = min(coalescing_dim_tile_size / vectorWidth,
                            available_threads)
remaining_threads = available_threads / threads_on_coalescing
```

If `remaining_threads > 1`, distribute them along the next dimension. Priority:
1. Next contiguous dimension of the dominant operand (for locality)
2. Other reduction dims (to reduce serial work)
3. Other parallel dims (to increase per-WG output)

Result as basis:
```
counts  = [threads_on_dim_X, ..., threads_on_coalescing_dim]
mapping = [iter_dim_X,       ..., coalescing_iter_dim]
```

The last position in counts always gets stride 1 → coalesced.

### Step 4: Subgroup distribution (subgroup_basis)

Subgroups can be distributed independently of threads. Options:

a. **Multiple subgroups along the coalescing dim** — increases the chunk
   processed per workgroup. Each subgroup handles a different slice,
   then subgroup-reduce if it's a reduction dim.

b. **Subgroups along a parallel dim** — each subgroup handles independent
   output. No inter-subgroup reduction.

c. **Single subgroup** — when there's enough parallel work to fill the GPU
   with workgroups.

Heuristic: if `parallelSize > numWGPs * numSIMDs`, use single subgroup per
workgroup. Otherwise, use multiple subgroups to increase per-WG throughput.

Comment: I think we need a better heuristic here. Basically, we need to balance occupany
here. We only want to use more subgroups per workgroup if the existing parallel
work (i.e. the number of workgroups we can launch) isn't enough to fill the
GPU. For example, if we only can launch 4 workgroups (not enough available
parallelism), we should use more subgroups on the coalescing dimension to
increase occupancy. But if we already have a lot of work available, there is no
need to use more subgroups. What I've done in past is just start assuming there
is a lot of parallel work, and keep increasing if the parallel work isn't
enough (increasing by multiplying by 2 usually is good).

### Step 5: Workgroup tile sizes

Tile parallel dims to create enough workgroups to fill the GPU:
```
parallel_tile = ceil(parallel_dim_size / target_num_workgroups)
```

For reduction dims: set to 0 (the full reduction is within the workgroup).
Exception: if the reduction is very large and we want to split across
workgroups (split-k style), tile it too.

Comment: Don't worry about split-k that is not our problem.

### Step 6: Partial reduction tile sizes

This controls the chunk size for the reduction loop. The chunk should be
large enough to give each thread enough work, but small enough to keep
register pressure manageable.

```
threads_on_reduction = (number of threads along reduction dims from Step 3)
partial_reduction_tile = threads_on_reduction * vectorWidth * batches_per_thread
```

Where `batches_per_thread` (target: 4-8) controls how many serial iterations
each thread does within a chunk. This becomes the `batch` dim in NestedLayout.

For parallel dims in the partial_reduction tile: 0 (no tiling).

### Step 7: Thread tile sizes (element_tile)

```
thread[coalescing_dim] = vectorWidth
thread[other_dims] = 1  (or 0 if not distributed at this level)
```

## How Existing Config Fields Map

No new fields needed. The existing fields express everything:

| Field | What it controls | How we set it |
|-------|-----------------|---------------|
| workgroup | Parallel dim tile sizes | From Step 5 |
| partial_reduction | Reduction loop chunk | From Step 6 |
| thread | Per-thread vector width | vectorWidth on coalescing dim |
| lane_basis | Thread distribution | From Step 3: coalescing dim gets lowest stride |
| subgroup_basis | Subgroup distribution | From Step 4 |

The key change is purely in how ReductionConfigUtils.cpp COMPUTES these values.

## Worked Examples

### Example 1: Innermost Reduction (softmax/layernorm)

```
linalg.generic : tensor<1024 x 4096 x f16> -> tensor<1024 x f16>
indexing_maps = [(d0,d1) -> (d0,d1), (d0,d1) -> (d0)]
iterator_types = [parallel, reduction]
```

**Step 1**: Dominant operand = input `tensor<1024x4096xf16>`
- Contiguous dim = dim 1 of operand
- indexing map: d1 → operand dim 1
- `coalescing_iter_dim = d1` (reduction dim)

**Step 2**: vectorWidth = 128/16 = 8, 4096 % 8 = 0 ✓

**Step 3**: subgroupSize = 64
- threads_on_d1 = min(4096/8, 64) = 64 (fills the subgroup)
- `lane_basis = [[64], [1]]` (64 threads on d1, stride 1)

**Step 4**: workgroupSize = 256 → 4 subgroups
- subgroups along d1 (more reduction bandwidth)
- `subgroup_basis = [[4], [1]]`

**Step 5**: `workgroup = [1, 0]` (1 row per WG)

**Step 6**: partial_reduction on d1:
- 4 subgroups × 64 threads × 8 elements = 2048 per chunk
- `partial_reduction = [0, 2048]`

**Step 7**: `thread = [0, 8]`

**Result**: Same as current config. Threads along reduction dim (d1),
which is also contiguous → coalesced ✓. Cooperative reduction via shuffles.

### Example 2: Outermost Reduction (column-wise sum)

```
linalg.generic : tensor<4096 x 1024 x f16> -> tensor<1024 x f16>
indexing_maps = [(d0,d1) -> (d0,d1), (d0,d1) -> (d1)]
iterator_types = [reduction, parallel]
```

**Step 1**: Dominant operand = input `tensor<4096x1024xf16>`
- Contiguous dim = dim 1 of operand
- indexing map: d1 → operand dim 1
- `coalescing_iter_dim = d1` (parallel dim!)

**Step 2**: vectorWidth = 8

**Step 3**: subgroupSize = 64
- threads_on_d1 = min(1024/8, 64) = 64
- No remaining threads
- `lane_basis = [[64], [1]]` (64 threads on d1)
- d1 is parallel → each thread handles independent columns

**Step 4**: single subgroup (1024 output elements → plenty of workgroups)
- `subgroup_basis = [[1], [1]]`

**Step 5**: workgroup tile for d1:
- Each WG handles 64 * 8 = 512 columns
- `workgroup = [0, 512]`
- 1024/512 = 2 workgroups (may want smaller tiles for more WGs)

**Step 6**: partial_reduction on d0:
- Each thread loops over d0 serially
- Tile d0 to limit vector size: e.g., 256 rows per chunk
- `partial_reduction = [256, 0]`

**Step 7**: `thread = [0, 8]` (or [1, 8])

**After tiling**: each chunk is `vector<256 x 512 x f16>`

**NestedLayout**:
```
subgroup_tile = [1, 1]
batch_tile    = [256, 1]   ← serial reduction per thread
thread_tile   = [1, 64]
element_tile  = [1, 8]
```
Check: 256 * 1 = 256 ✓, 1 * 64 * 8 = 512 ✓

**Reduction behavior**: dim 0 has thread_tile=1, subgroup_tile=1 → NO shuffle.
The 256 rows are reduced locally in each thread's batch dim. Each thread
accumulates 256 values for each of its 8 columns. Final result: each thread
holds the partial sum for its 8 columns.

**Memory access**: At each row, 64 threads read 8 contiguous f16 values each
→ 512 contiguous elements → perfectly coalesced ✓

**This is fundamentally different from the current config**, which would try
to put threads along d0 (strided access) and shuffle-reduce across them.

### Example 3: Batch Norm (multi-dim reduction)

```
linalg.generic : tensor<64 x 256 x 56 x 56 x f16> -> tensor<256 x f16>
indexing_maps = [(d0,d1,d2,d3) -> (d0,d1,d2,d3),  // input
                 (d0,d1,d2,d3) -> (d1)]             // output
iterator_types = [reduction, parallel, reduction, reduction]
```

**Step 1**: Dominant operand = input
- Contiguous dim = dim 3 (W) of operand
- indexing map: d3 → operand dim 3
- `coalescing_iter_dim = d3` (reduction dim)

**Step 2**: vectorWidth = 8, but 56 % 8 = 0 ✓

**Step 3**: subgroupSize = 64
- threads_on_d3 = min(56/8, 64) = 7 → not power of 2
- Try vectorWidth = 4: threads_on_d3 = 56/4 = 14 → 64/14 doesn't divide
- Try vectorWidth = 2: threads_on_d3 = 56/2 = 28 → 64/28 doesn't divide
- Try vectorWidth = 1: threads_on_d3 = 56 → 64/56 doesn't divide

None divide cleanly. Options:
a) Use 56 threads on d3 (underutilize: 56 of 64 threads active)
b) Pad to 64 with masking
c) Spread to another dim: e.g., 8 threads on d3 (vecWidth=8), but 56%8=0,
   64/8 = 8 remaining threads → put on d2 (H, also reduction, size 56)
   threads_on_d2 = 8 (with 56/8=7 per thread serial → need 56%8=0 ✓)

Going with (c): 8 threads on d3 (vecWidth=7... no that's not right)

Actually, let me reconsider. vecWidth=8, but 56/8 = 7. We need 56 / (threads * vecWidth) to be integer.
- vecWidth=8, threads=7: 7*8=56 ✓, but 7 threads doesn't divide 64
- vecWidth=4, threads=14: 14*4=56 ✓, remaining=64/14... nope
- vecWidth=2, threads=28: 28*2=56 ✓, remaining=64/28... nope
- vecWidth=1, threads=56: remaining=64/56... nope

So 56 is tricky. In practice, batch norm spatial dims are often handled
differently (e.g., flatten H*W = 3136 first, or tile W to a power of 2).

With tiling: tile W to 32 (or 16):
- workgroup tile includes spatial tiling
- W_tile = 32, vecWidth = 8, threads_on_d3 = 32/8 = 4
- remaining = 64/4 = 16 → put on d2 (H): threads_on_d2 = min(56,16) = 16
  But 56 % 16 ≠ 0... tile H too? H_tile = 16.
- Or just use fewer threads: 4 threads on d3, that's it.
  subgroupSize threads = 64, only using 4 → wasteful

Alternative: flatten spatial dims conceptually, or tile W differently.

This case highlights that **batch norm with non-power-of-2 spatial dims needs
careful tiling before thread distribution**. The config generator should:
1. First decide on tiles that produce power-of-2 (or at least subgroup-friendly)
   inner dimensions
2. Then distribute threads

For the design, the algorithm handles this by adjusting vectorWidth and accepting
that not all threads may be active. We could also add a preliminary step that
finds good workgroup tiles for the spatial dims.

### Example 4: Argmax

```
linalg.generic : tensor<1024 x 4096 x f16> -> tensor<1024 x f16>, tensor<1024 x i64>
indexing_maps = [(d0,d1) -> (d0,d1), (d0,d1) -> (d0), (d0,d1) -> (d0)]
iterator_types = [parallel, reduction]
```

Two outputs: values and indices. This is a multi-output reduction with a
custom combiner: `if new_val > old_val: update both val and idx`.

**Coalescing analysis is identical to Example 1**: threads along d1 (contiguous
reduction dim).

**The difference is the combiner**: current code requires `checkSingleCombiner`
to pass. For argmax, we'd need to relax this to allow multi-output reductions
where the combiners are coupled.

The thread/subgroup reduction would use a custom shuffle pattern:
```
(val, idx) = shuffle_reduce((val, idx), custom_argmax_combine)
```

**Config-wise**, the layout is the same as Example 1. The only change needed
is supporting the multi-output combiner in the reduction distribution pattern,
not in the config generation.

### Example 5: Scan/Cumsum (future)

```
input:  tensor<1024 x 4096 x f16>
output: tensor<1024 x 4096 x f16>
scan along dim 1 (contiguous)
```

**Coalescing**: Same as reduction — threads along d1 (contiguous).

**Thread distribution**: Same basis as Example 1.

**Difference**: Instead of reducing to a scalar, scan produces a full output.
The pipeline would need a scan-specific distribution pattern:
1. Each thread scans its local elements sequentially
2. Cross-thread parallel scan via shuffles (Hillis-Steele / Blelloch)
3. Broadcast prefix back to update local values

The config can use the same lane_basis. The distinction between reduce vs
scan is in the distribution pattern, not the config.

## What Changes in Code

### ReductionConfigUtils.cpp (main change)

Replace the current logic that always targets the last reduction dim with:

```cpp
// 1. Find coalescing iteration dim
int64_t coalescingIterDim = findCoalescingDim(op);

// 2. Determine vector width
int64_t vectorWidth = computeVectorWidth(op, coalescingIterDim, target);

// 3. Build thread basis with coalescing dim last (stride 1)
Basis laneBasis = buildThreadBasis(op, coalescingIterDim, vectorWidth,
                                    subgroupSize);

// 4. Build subgroup basis
Basis subgroupBasis = buildSubgroupBasis(op, laneBasis, target);

// 5. Set tile sizes
// - workgroup: tile parallel dims
// - partial_reduction: based on thread/subgroup counts
// - thread: vectorWidth on coalescing dim
```

### New helper: findCoalescingDim

```cpp
static int64_t findCoalescingDim(linalg::LinalgOp op) {
  // Find the dominant operand (largest input by estimated size)
  Value dominant = findDominantOperand(op);
  auto tensorType = cast<RankedTensorType>(dominant.getType());

  // Innermost dim of the operand tensor is contiguous
  int64_t contiguousOperandDim = tensorType.getRank() - 1;

  // Map back through indexing map to iteration space
  OpOperand *operand = /* find operand for dominant value */;
  AffineMap map = op.getMatchingIndexingMap(operand);

  // Find which iteration dim maps to contiguousOperandDim
  for (int64_t i = 0; i < map.getNumDims(); ++i) {
    auto expr = map.getResult(contiguousOperandDim);
    if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
      return dimExpr.getPosition();
    }
  }
  // Fallback: last dim
  return op.getNumLoops() - 1;
}
```

### LLVMGPUConfigureTensorLayouts.cpp

No changes needed. The existing `setGPULoweringConfigLayout` already consumes
lane_basis and subgroup_basis generically. The coalescing-aware basis from
the config generator flows through unchanged.

### GPUNestedLayoutDistributionPatterns.cpp

No changes needed for basic reductions. The `DistributeMultiReduction`
pattern already handles:
- Thread-local reduction (batch dims)
- Shuffle reduction (when threadTile[dim] > 1 for reduction dims)
- Subgroup reduction (when subgroupTile[dim] > 1 for reduction dims)

When threads are along a parallel dim and the reduction dim has
threadTile=1, the reduction is entirely thread-local. This already works.

### Future: Scan/Sort distribution patterns

New patterns would be needed in GPUNestedLayoutDistributionPatterns.cpp for
scan and sort operations, but the config generation and layout assignment
use the same infrastructure.

## Open Questions

1. **Multiple operands with conflicting coalescing needs**: When operands
   have different indexing maps, which one wins? Current proposal: dominant
   operand (largest input). Could also consider shared memory promotion for
   the losing operand.

2. **Non-power-of-2 sizes**: How to handle dims like 56 (batch norm spatial)?
   Options: masking, preliminary tiling, accepting underutilization.

3. **Dynamic shapes**: When the coalescing dim size is unknown at compile time,
   we can't guarantee divisibility. May need to pick conservative vector widths
   or use masking.

4. **Interaction with fused ops**: A dispatch may have multiple ops with
   different indexing maps. The shared workgroup tiles and thread distribution
   must be compatible across all ops. Current code handles this via
   `sharedWgpTiles` — same principle applies.

5. **Should we rename the function?** `setReductionConfig` →
   `setMemoryBoundConfig` to reflect the broader scope.

6. **Should we add a "coalesced" field to the config?** Not strictly needed
   (the basis encodes the same info), but could aid debuggability:
   ```mlir
   #iree_gpu.lowering_config<{
     ...,
     coalesced = [operand_index, operand_dim, iter_dim]
   }>
   ```
