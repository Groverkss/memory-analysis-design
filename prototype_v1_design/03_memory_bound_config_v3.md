# Design v3: Memory-Bound Operation Configuration

## The Right Order of Operations

```
1. Parallelism Analysis    → what can be parallelized, how dims relate
2. Workgroup Tiling        → how to split work, per-WG tile shapes
3. Subgroup Scaling        → workgroup size based on available parallelism
4. Thread Layout           → coalescing within each WG's tile
```

Everything flows from Step 1. Coalescing is a later concern that operates
on whatever tile each workgroup receives.

---

## Step 1: Parallelism Analysis

### 1a. Identify fully parallel dimensions

A dimension `d` is fully parallel if:
- It has `parallel` iterator type (not reduction)
- Tiling along `d` produces independent workgroups

For each parallel dimension, determine which operands it affects:

```
For each parallel iter_dim d:
  For each operand:
    If d appears in the operand's indexing map results:
      → parallelizing d SHRINKS this operand's per-WG tile
    If d does NOT appear:
      → parallelizing d has NO EFFECT on this operand (broadcast)
```

Example — layer norm: `tensor<M x N> → tensor<M>`, reduce N
```
d0 = parallel, d1 = reduction
indexing_maps: input (d0,d1)→(d0,d1), output (d0,d1)→(d0)

Parallelizing d0:
  input:  d0 appears → shrinks (M/tile × N per WG)
  output: d0 appears → shrinks (M/tile per WG)
```

Example — batch norm: `tensor<N x C x H x W> → tensor<C>`, reduce N,H,W
```
d0=N(red), d1=C(par), d2=H(red), d3=W(red)
indexing_maps: input (d0,d1,d2,d3)→(d0,d1,d2,d3), output (d0,d1,d2,d3)→(d1)

Parallelizing d1 (C):
  input:  d1 appears → shrinks (N × C/tile × H × W per WG)
  output: d1 appears → shrinks (C/tile per WG)
```

Example — broadcast add fused with reduction:
```
input:  (d0,d1) → (d0,d1)    tensor<M x N>
bias:   (d0,d1) → (d1)       tensor<N>
output: (d0,d1) → (d0)       tensor<M>
d0 = parallel, d1 = reduction

Parallelizing d0:
  input:  d0 appears → shrinks
  bias:   d0 NOT in map → unchanged (broadcast)
  output: d0 appears → shrinks
```

### 1b. Dimension relationships

Some dimensions are **linked** — parallelizing one implies something about
another.

**Direct linkage**: Two parallel dims that appear together in the same operand.
Parallelizing one without the other means the operand's tile is still large
in the other dim.

**Broadcast linkage**: If a parallel dim appears in some operands but not
others, parallelizing it reduces work for some operands but not all. The
non-affected operand might become the bottleneck.

**Reduction-parallel linkage**: Not directly relevant for parallelism, but
important for understanding total work: the reduction dims determine how much
work each WG does regardless of parallel tiling.

### 1c. Total parallelism

```
total_parallelism = product of all parallel dim sizes
```

This is the maximum number of independent workgroups we could launch. In
practice, we tile to get fewer, larger workgroups.

### 1d. Per-operand work estimate

For each operand, estimate the total elements accessed:
```
operand_work = product of dim sizes that appear in the operand's indexing map
```

After parallelizing a dim `d` with tile size `T`:
```
operand_work_per_wg = operand_work / T  (if d appears in operand's map)
                    = operand_work      (if d doesn't appear)
```

This tells us which operands are "large" (memory-bound) per workgroup vs
"small" (fits in cache/registers). A small operand doesn't need coalescing
optimization — it's cheap to load however.

---

## Step 2: Workgroup Tiling

### 2a. Goal

Find workgroup tile sizes for the parallel dimensions such that:
1. Enough workgroups to fill the GPU
2. Per-WG work is balanced (not too much, not too little)
3. Operand tiles are manageable

### 2b. Simple heuristic

Start with a target number of workgroups:
```
target_wgs = numWGPs * numSIMDs  // enough to fill the GPU
// or some multiple for latency hiding
```

Then choose parallel tile sizes to produce roughly `target_wgs` workgroups:
```
total_parallelism = product(parallel_dim_sizes)
target_tile_product = total_parallelism / target_wgs

Distribute across parallel dims, prioritizing:
- Dims that appear in more operands (tiling them reduces more work)
- Larger dims (more room to tile)
```

### 2c. Dimension-aware tiling

Consider the effect on each operand:

Example — bias + reduce:
```
input:  tensor<1024 x 4096>  — 4M elements
bias:   tensor<4096>          — 4K elements
output: tensor<1024>          — 1K elements

Only d0 is parallel. Parallelizing d0 with tile=1:
  input per WG: 1 × 4096 = 4096 elements
  bias per WG:  4096 (unchanged — broadcast)
  output per WG: 1
  WGs: 1024
```

The bias is small (4K elements) and doesn't change with d0 tiling. Not worth
worrying about its coalescing — it'll fit in L2 cache after the first WG
loads it.

If d0 were tiled more coarsely (tile=64):
```
  input per WG: 64 × 4096 = 256K elements — this is the bottleneck
  bias per WG:  4096 — still tiny
  WGs: 16
```

### 2d. Output: per-WG tile shapes

After workgroup tiling, we know each operand's per-WG tile shape.
```
For each operand:
  per_wg_shape = operand_shape with parallel dims divided by their tiles
```

These shapes are what the within-WG analysis (steps 3-4) operates on.

---

## Step 3: Subgroup Scaling

After workgroup tiling, check if we have enough parallelism:

```
num_wgs = total_parallelism / product(parallel_tiles)

subgroups_per_wg = 1
while num_wgs * subgroups_per_wg < target_occupancy:
    subgroups_per_wg *= 2
    if subgroups_per_wg * subgroupSize > maxWorkgroupSize:
        subgroups_per_wg /= 2
        break

workgroupSize = subgroups_per_wg * subgroupSize
```

More subgroups per WG means each WG processes more data (along the reduction
or parallel dims within the WG), but we have fewer WGs. This is only
beneficial when we don't have enough WGs to fill the GPU.

---

## Step 4: Thread Layout (Coalescing)

Now we know:
- Each WG's tile shapes for each operand
- The workgroup size (number of threads)
- Which dims are reduction (serial/distributed) vs parallel (independent)

### 4a. Per-operand coalescing analysis

For each operand's per-WG tile:
- Identify contiguous dims (innermost first)
- Compute ideal thread distribution for coalesced access
- Note which iteration dims this corresponds to

### 4b. Build the lane_basis

Since we specify coalescing per-operand (per your comment), the config
should carry per-operand coalescing info. ConfigureTensorLayouts uses
this to set appropriate layouts and shared memory conversions.

However, we still need a single lane_basis for the iteration space (since
the tiling is on the iteration space). The lane_basis is chosen to best
serve the most memory-intensive operand, and operands that disagree get
shared memory conversions.

**Thread filling order**: From the dominant operand's innermost dim outward,
mapping through the indexing map to iteration dims.

```
vectorWidth = maxLoadBits / bitWidth  (always max, masking handles the rest)

remaining_threads = subgroupSize
For each dim of dominant operand (innermost first):
  iter_dim = indexing_map_inverse(operand_dim)
  dim_size = per_wg_tile_size[iter_dim]
  threads_here = min(ceil(dim_size / vecWidth_for_this_dim), remaining_threads)
  assign threads_here to iter_dim
  remaining_threads /= threads_here
  if remaining_threads <= 1: break
```

vecWidth applies to the innermost dim only; other dims get thread granularity 1.

### 4c. Partial reduction tile

The partial_reduction tile controls the reduction loop chunk size. It should
be set so the vectorized body has a reasonable size:

```
For distributed reduction dims: chunk = threads_on_dim * vectorWidth
For serial reduction dims: chunk = target_batch_per_thread (e.g., 4-8)
For parallel dims: 0 (no tiling at this level)
```

---

## Concrete Examples

### Matvec: tensor<4 x 4096 x f16> × tensor<4096 x 4096 x f16> → tensor<4 x 4096 x f16>

This is a contraction with iteration space (M, N, K) = (4, 4096, 4096),
iterator_types = [parallel, parallel, reduction].

```
lhs: (d0,d1,d2) → (d0,d2)   tensor<4 x 4096 x f16>      (skinny matrix)
rhs: (d0,d1,d2) → (d2,d1)   tensor<4096 x 4096 x f16>    (the big matrix)
out: (d0,d1,d2) → (d0,d1)   tensor<4 x 4096 x f16>
```

**Step 1**: d0=M(par)=4, d1=N(par)=4096, d2=K(red)=4096.
- total_parallelism = 4 × 4096 = 16384
- d0 is parallel but small (M=4)
- d1 is the large parallel dim

Per-operand analysis:
- lhs: dims are (d0, d2) → parallelizing d1 does NOT shrink lhs. It's
  broadcast across N. Per-WG lhs is always 4 × 4096 = 16K elements.
- rhs: dims are (d2, d1) → parallelizing d1 SHRINKS rhs.
  Full rhs = 4096 × 4096 = 16M elements.
  Parallelizing d0 does NOT shrink rhs (d0 not in rhs map). Rhs is
  broadcast across M.
- out: dims are (d0, d1) → parallelizing either d0 or d1 shrinks output.

Dimension relationships:
- d0 and d1 are both parallel but affect different operands:
  - d0 shrinks lhs and out, NOT rhs
  - d1 shrinks rhs and out, NOT lhs
- rhs is the dominant operand (16M elements). Only d1 reduces rhs traffic.
- lhs is modest (16K elements). Only d0 reduces lhs traffic, but it's
  already small enough that L2 cache handles it.

**Step 2**: tile_d1 to create enough WGs.
- tile_d0 = 1, tile_d1 = 128: WGs = 4 × 32 = 128.
  Per-WG rhs = 4096 × 128 = 512K elements = 1MB.
  Per-WG lhs = 1 × 4096 = 4K elements (broadcast, same for all d1 WGs).
  Per-WG out = 1 × 128 = 128 elements.

- tile_d0 = 4, tile_d1 = 128: WGs = 1 × 32 = 32.
  Per-WG rhs = 4096 × 128 = 1MB (same — d0 doesn't affect rhs).
  Per-WG lhs = 4 × 4096 = 32KB.
  Per-WG out = 4 × 128 = 512 elements.

  Fewer WGs but each WG computes 4 output rows sharing the same rhs slice.
  This is better if we can keep 4 lhs rows in registers simultaneously.

Key observation: **rhs is broadcast across d0, lhs is broadcast across d1**.
Tiling d0 doesn't reduce rhs traffic. Tiling d1 doesn't reduce lhs traffic.
The optimal WG tile should tile d1 aggressively (to reduce per-WG rhs) and
keep d0 intact or lightly tiled (lhs is small, and sharing rhs across M rows
is beneficial).

**Step 3**: With tile_d0=4, tile_d1=128: 32 WGs.
- If target occupancy = 512, that's too few → increase subgroups.
- 2 subgroups: workgroupSize=128. Still only 32 WGs.
- 4 subgroups: workgroupSize=256. 32 WGs × 4 subgroups = 128 SIMDs active.
  Closer but may still want more.
- Or: reduce tile_d1 to 64 → 64 WGs. Each WG: rhs = 4096×64 = 256K = 512KB.
  With 2 subgroups: 64 WGs × 2 = 128 SIMDs.

**Step 4**: Per-WG tiles (tile_d0=4, tile_d1=128):
- lhs per WG: tensor<4 × 4096> (d0=4, d2=4096)
- rhs per WG: tensor<4096 × 128> (d2=4096, d1=128)
- out per WG: tensor<4 × 128>

Coalescing analysis:
- rhs: contiguous dim is dim 1 → d1 in iteration space.
  Per-WG d1 size = 128. threads on d1 = 128/8 = 16.
  remaining = 64/16 = 4. Can put on d2 (K, reduction) or d0 (M).
- lhs: contiguous dim is dim 1 → d2 in iteration space.
  Wants threads on d2. But rhs wants threads on d1.
  → Conflict. VectorDistribute handles via shared memory for one of them.

Thread distribution favoring rhs (dominant):
- threads on d1 = 16 (rhs coalesced)
- threads on d2 = 4 (reduction cooperation)
- lane_basis = [[4, 16], [2, 1]]

Each K-step: 16 threads each load 8 contiguous f16 along d1 from rhs →
coalesced ✓. Same 16 threads each load from lhs, but lhs has different
contiguous dim → needs shared memory or accepts strided access. Since
lhs per WG is only 32KB, shared memory promotion works well.

The M=4 dim is handled as batch: each thread computes 4 output rows (or
threads split across M too if beneficial). Since M=4 is small, it naturally
becomes part of the batch/outer tile in NestedLayout.

Possible config:
```
workgroup = [4, 128, 0]
partial_reduction = [0, 0, 256]
thread = [0, 8, 1]
lane_basis = [[4, 16], [2, 1]]   // 4 on K, 16 on N
subgroup_basis = [[1, 1], [2, 1]]
```

NestedLayout for vector<4 × 128 × ... (after K tiling)>:
- M=4 is in batch (serial per thread, all 4 rows)
- N=128 is distributed: 16 threads × 8 elements
- K=256 (chunk): 4 threads × batch_k × 1 element

The general framework handles matvec without any special case:
- Parallelism analysis reveals the broadcast structure (lhs↔d1, rhs↔d0)
- Workgroup tiling respects which dims reduce which operands' traffic
- Thread layout coalesces the dominant operand (rhs)
- Conflicts (lhs wants different layout) resolved by shared memory

### Batched Matvec: tensor<B x 4 x K x f16> × tensor<B x K x N x f16>

Same as above but with a batch dim. d_batch is parallel — each WG handles
one (or a few) batch elements. The within-WG analysis is identical to the
non-batched case.

The broadcast analysis extends: lhs is broadcast across N, rhs is broadcast
across M. Batch dim parallelizes all operands (appears in all indexing maps).

---

### Layer Norm: tensor<1024 x 4096 x f16> → tensor<1024>

**Step 1**: d0=parallel, d1=reduction.
- total_parallelism = 1024
- Parallelizing d0: input shrinks (per WG: tile_d0 × 4096), output shrinks

**Step 2**: target_wgs ≈ 512 (say).
- tile_d0 = ceil(1024/512) = 2
- Per-WG: input tile = 2 × 4096 = 8K elements, output tile = 2 elements
- Or tile_d0 = 1: 1024 WGs, input = 1×4096 per WG

**Step 3**: 1024 WGs > target → 1 subgroup per WG.

**Step 4**: Per-WG input tile = tensor<1 x 4096>.
- Contiguous: dim 1 (d1). vectorWidth = 8.
- threads on d1 = min(4096/8, 64) = 64
- lane_basis = [[64], [1]]

```
workgroup = [1, 0]
partial_reduction = [0, 2048]  // or full 4096 if 1 subgroup
thread = [0, 8]
lane_basis = [[64], [1]]
subgroup_basis = [[1], [1]]
```

### Column-wise Sum: tensor<4096 x 1024 x f16> → tensor<1024>

**Step 1**: d0=reduction, d1=parallel.
- total_parallelism = 1024
- Parallelizing d1: input shrinks (4096 × tile_d1), output shrinks (tile_d1)

**Step 2**: tile_d1 = 128 → 8 WGs.
- Per-WG: input tile = tensor<4096 × 128>, output = tensor<128>

**Step 3**: 8 WGs is low. Scale subgroups.
- subgroups_per_wg = 1: 8 WGs. Not enough if target = 512.
- But adding subgroups doesn't help here — they'd increase per-WG work,
  reducing WGs further.
- Actually, subgroups operate within the WG on the SAME tile. They don't
  change the number of WGs.
- 8 WGs might be fine if each WG has enough memory traffic to saturate
  bandwidth. 4096 × 128 × 2 bytes = 1MB per WG — that's a LOT of data.
  Memory bandwidth will be saturated with just a few WGs.
- Keep 1 subgroup.

Alternative: tile_d1 = 512 → 2 WGs. Even 2 WGs with 1MB+ each is fine.

**Step 4**: Per-WG input tile = tensor<4096 × 128>.
- Contiguous: dim 1 (d1, parallel). vectorWidth = 8.
- threads on d1 = min(128/8, 64) = 16
- remaining = 64/16 = 4. Put on d0 (reduction):
  threads on d0 = min(4096, 4) = 4
- lane_basis = [[4, 16], [0, 1]]

```
workgroup = [0, 512]
partial_reduction = [256, 0]  // chunk of d0 per iteration
thread = [1, 8]
lane_basis = [[4, 16], [0, 1]]
subgroup_basis = [[1, 1], [0, 1]]
```

### Batch Norm: tensor<64 x 256 x 56 x 56 x f16> → tensor<256>

**Step 1**: d0=N(red), d1=C(par), d2=H(red), d3=W(red).
- total_parallelism = 256 (just C)
- Parallelizing d1: input = N×(C/tile)×H×W per WG, output = C/tile per WG

**Step 2**: tile_d1 = 1 → 256 WGs.
- Per-WG: input = 64 × 1 × 56 × 56 = 200704 elements per WG
- That's 400KB at f16. Substantial.

**Step 3**: 256 WGs. If target ≈ 512, maybe 2 subgroups per WG.
- workgroupSize = 128. Or just 1 subgroup if 256 is close enough.

**Step 4**: Per-WG input tile = tensor<64 × 1 × 56 × 56>.
- Contiguous: dim 3 (d3, W=56). vectorWidth = 8.
- threads on d3 = ceil(56/8) = 7 → use 8 with masking
- remaining = 64/8 = 8. Put on d2 (H=56):
  threads on d2 = 8
- remaining = 1.
- lane_basis = [[8, 8], [2, 3]]

```
workgroup = [0, 1, 0, 0]
partial_reduction = [1, 0, 56, 56]  // or tiled smaller
thread = [0, 0, 1, 8]
lane_basis = [[8, 8], [2, 3]]
subgroup_basis = [[1, 1], [2, 3]]
```

Note how the per-WG analysis naturally handles the "big picture":
- d1=C is the parallel dim → workgroup tiling
- d0=N is a large serial reduction → loop per thread
- d2=H, d3=W are spatial → threads distributed across them for coalescing
- After parallelizing d1, the per-WG input is 64×1×56×56 which reveals
  that d0 (N) is the big serial dim and d2,d3 are where threads should go

---

## Summary

The design has four phases:

1. **Parallelism analysis**: Understand which dims are parallel, how they
   relate across operands, total available parallelism.

2. **Workgroup tiling**: Choose parallel tiles to create enough WGs. This
   determines per-WG operand tile shapes and reveals which operands are
   large (need coalescing) vs small (can ignore).

3. **Subgroup scaling**: If not enough WGs, increase subgroups per WG.
   Start at 1, double until sufficient.

4. **Thread layout**: Given per-WG tiles, distribute threads for coalesced
   access on the largest operands. Coalescing is per-operand, specified in
   the config. VectorDistribute handles conflicts via shared memory.

The key insight is that **parallelism analysis comes first** because it
determines what each workgroup sees, which in turn determines what needs
coalescing and what doesn't.
