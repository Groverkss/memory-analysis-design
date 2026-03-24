# Design: Workgroup Tiling Heuristic

## Input

From the parallelism analysis we have:
- A set of **dimension groups**, each with:
  - A symbolic name (P0, P1, ... for parallel; R0, R1, ... for reduction)
  - A size (static or dynamic)
  - Whether it's parallelizable
- The **dispatch DAG** showing how tensors flow between ops
- Per-tensor shapes expressed in terms of these groups

## Goal

Decide how to tile the parallelizable groups across workgroups. After tiling
we need to know:
1. **Workgroup tile sizes** for each parallel group (how much each WG processes)
2. **Number of workgroups** that will be launched
3. **Per-WG operand tile shapes** for each tensor in the dispatch

The tiling must balance:
- Enough workgroups to fill the GPU
- Each WG has enough work to amortize overhead
- Per-WG operand tiles are manageable (not too large for registers/cache)

## Hardware Parameters

From `TargetAttr` / `TargetWgpAttr` / `TargetChipAttr`:
```
subgroupSize      = e.g., 64 (AMD) or 32 (NVIDIA)
maxWorkgroupSize  = e.g., 1024
numWGPs           = e.g., 304 (gfx942) or 512 (default)
numSIMDs          = e.g., 4 (SIMDs per WGP)
```

Target occupancy (minimum workgroups to saturate the GPU):
```
targetOccupancy = numWGPs * numSIMDs  // e.g., 304 * 4 = 1216
```

## The Heuristic

### Principle: Start with minimum WG tile (tile=1), then coarsen if too many WGs

For memory-bound operations, each WG should process enough data to
saturate memory bandwidth. But we also need enough WGs to fill the GPU.

The simplest approach: tile each parallel group to 1 (maximum parallelism),
then coarsen if we have way too many WGs.

### Step 1: Compute maximum parallelism

```
maxParallelism = product of all parallelizable group sizes
```

For the layer norm example:
```
P0 = 2, P1 = 32
maxParallelism = 2 * 32 = 64
```

### Step 2: Start with tile=1 for all parallel groups

```
wg_tile[group] = 1 for each parallelizable group
numWGs = maxParallelism  // = 64 for layer norm
```

### Step 3: Check if we have enough parallelism

```
if numWGs >= targetOccupancy:
    // Plenty of WGs. Each WG does minimal parallel work.
    // This maximizes parallelism. Good.
    done.
```

For layer norm: 64 < 1216 → NOT enough.

### Step 4: If not enough WGs, we need more subgroups per WG

When `numWGs < targetOccupancy`, we can't create more WGs (we've already
tiled to 1). Instead, use more subgroups per WG to increase occupancy.

```
subgroupsPerWG = 1
while numWGs * subgroupsPerWG < targetOccupancy:
    subgroupsPerWG *= 2
    if subgroupsPerWG * subgroupSize > maxWorkgroupSize:
        subgroupsPerWG /= 2
        break
```

For layer norm: 64 WGs, target 1216.
- subgroupsPerWG = 2: 64*2 = 128. Still < 1216.
- subgroupsPerWG = 4: 64*4 = 256. Still < 1216.
- subgroupsPerWG = 8: 64*8 = 512. Still < 1216.
- subgroupsPerWG = 16: 64*16 = 1024. Close to 1216. And 16*64 = 1024 = maxWorkgroupSize.

So workgroupSize = 1024, which is what the current config also produces
for this case.

### Step 5: If too many WGs, coarsen the tiles

When `numWGs >> targetOccupancy` (say 10x or more), we waste launch
overhead. Coarsen the smallest parallel groups first:

```
while numWGs > coarsenThreshold * targetOccupancy:
    // Find the smallest parallelizable group that hasn't been fully coarsened
    group = smallest_parallel_group_by_size
    // Double its tile (halve the WGs along this dim)
    wg_tile[group] *= 2
    numWGs /= 2
```

This is less common for memory-bound ops (they usually have limited
parallelism), but matters for cases like outermost reduction where the
parallel dim is large.

Example — outermost reduction: `tensor<4096 x 1024>`, reduce dim 0.
```
P0 = 1024 (parallel)
maxParallelism = 1024
targetOccupancy = 1216

numWGs = 1024. Close enough to target — no coarsening needed.
subgroupsPerWG = 2: 1024*2 = 2048 > 1216. Sufficient.
workgroupSize = 128.
Per-WG: each WG handles 1 column of 4096 rows.
```

But if P0 were 100000:
```
numWGs = 100000. Way too many.
Coarsen: wg_tile[P0] = 2 → 50000. Still too many.
Continue until numWGs ≈ targetOccupancy.
wg_tile[P0] = 128 → 781 WGs. Close to 1216.
Per-WG: each WG handles 128 columns.
```

### Step 6: Handle multiple parallel groups

When there are multiple parallel groups (like layer norm with P0=2, P1=32),
the tiling interacts:

```
numWGs = product(size[g] / wg_tile[g] for g in parallel_groups)
```

To coarsen, we choose which group to coarsen. The priority should be:
- Coarsen groups that affect MORE operands (bigger per-WG reduction)
- Coarsen larger groups first (more room to tile)
- Keep the product close to targetOccupancy

In practice, for memory-bound ops, the parallel groups are usually small
enough that we DON'T need to coarsen — we need MORE parallelism (step 4).

### Step 7: Record per-WG operand shapes

After deciding wg_tiles for each parallel group, compute per-WG shapes:

```
For each tensor:
  per_wg_shape[dim] = tensor_shape[dim]                    if dim's group is reduction
  per_wg_shape[dim] = tensor_shape[dim] / wg_tile[group]   if dim's group is parallel
```

This is what the coalescing analysis (next phase) operates on.

## Worked Examples

### Layer Norm: P0=2, P1=32, R0=16384, R1=10

```
maxParallelism = 2 * 32 = 64
numWGs = 64 (tile all to 1)
targetOccupancy = 1216

64 < 1216 → need subgroup scaling
subgroupsPerWG = 16 (16*64=1024, 64*16=1024)
workgroupSize = 1024

Per-WG operand shapes:
  input  <P0xP1xR1xR0xf16> → <1x1x10x16384xf16>  = 160K elements
  mean   <P0xP1xf32>        → <1x1xf32>            = scalar
  var    <P0xP1xf32>        → <1x1xf32>            = scalar
  output <P0xP1xR1xR0xf32>  → <1x1x10x16384xf32>  = 160K elements
```

Observation: mean and var are scalars per WG. Only input/output matter
for coalescing.

### Softmax: P0=512, R0=10240

```
maxParallelism = 512
numWGs = 512 (tile P0 to 1)
targetOccupancy = 1216

512 < 1216 → need subgroup scaling
subgroupsPerWG = 4 (4*64=256, 512*4=2048 > 1216)
workgroupSize = 256

Per-WG operand shapes:
  input  <P0xR0xf32> → <1x10240xf32> = 10240 elements
  max    <P0xf32>    → <1xf32>        = scalar
  sum    <P0xf32>    → <1xf32>        = scalar
  output <P0xR0xf32> → <1x10240xf32> = 10240 elements
```

### Outermost Reduction: P0=1024, R0=4096

```
maxParallelism = 1024
numWGs = 1024 (tile P0 to 1)
targetOccupancy = 1216

1024 ≈ 1216 → close enough, maybe 2 subgroups to be safe
subgroupsPerWG = 2 (2*64=128, 1024*2=2048)
workgroupSize = 128

Per-WG operand shapes:
  input  <R0xP0xf16> → <4096x1xf16> = 4096 elements
  output <P0xf16>    → <1xf16>       = 1 element

Hmm, each WG processes only 1 column (4096 elements). That's 8KB at f16.
Quite small. Could coarsen P0 to bundle more columns per WG:

Alternative: wg_tile[P0] = 8 → 128 WGs
subgroupsPerWG = 8 → 128*8 = 1024
workgroupSize = 512

Per-WG: input = <4096x8xf16> = 32K elements = 64KB.
Better — each WG has more data to process.
```

### Matvec: P0=4, P1=4096, R0=4096

```
maxParallelism = 4 * 4096 = 16384
numWGs = 16384 (tile all to 1)
targetOccupancy = 1216

16384 >> 1216 → too many WGs, coarsen

Coarsen P1 (size=4096, larger, affects rhs):
wg_tile[P1] = 2  → numWGs = 4*2048 = 8192
wg_tile[P1] = 4  → numWGs = 4*1024 = 4096
wg_tile[P1] = 8  → numWGs = 4*512  = 2048
wg_tile[P1] = 16 → numWGs = 4*256  = 1024 ≈ 1216. Stop.

So wg_tile = [1, 16] → 1024 WGs.
subgroupsPerWG = 2: 1024*2 = 2048. Enough.
workgroupSize = 128.

Per-WG operand shapes:
  lhs <P0xR0xf16>    → <4x4096xf16>  = 16K elements (broadcast across P1)
  rhs <R0xP1xf16>    → <4096x16xf16> = 64K elements
  out <P0xP1xf16>    → <4x16xf16>    = 64 elements
```

## What This Produces for the Config

```
workgroup = [wg_tile_for_each_op_dim]   // mapped through indexing maps
```

The workgroup tile for an op is determined by mapping the group-level tiles
to the op's iteration space. For groups that the op doesn't touch (broadcast),
the tile is 0 (no tiling on that dim).

## Summary

The heuristic is:
1. **Start with max parallelism** (tile=1 for all parallel groups)
2. **Scale subgroups** if not enough WGs
3. **Coarsen** if too many WGs (prefer larger groups, ones that affect more operands)
4. **Record per-WG shapes** for the coalescing phase

The key insight is that for memory-bound ops, we almost always have
LESS parallelism than needed (steps 2-3 are the common path). The
coarsening path (step 5) mainly applies to matvec-like cases where
one parallel dim is very large.
