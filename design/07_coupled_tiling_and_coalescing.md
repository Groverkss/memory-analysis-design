# Design: Coupled Workgroup Tiling + Coalescing

## The Problem with Sequential Design

v3 and the workgroup heuristic doc treated tiling and coalescing as
sequential: first tile for parallelism, then figure out coalescing on the
per-WG tile. But they're coupled:

**The contiguous dimension might be parallelizable.** If we tile it all the
way out (tile=1), there's nothing left within the WG for threads to coalesce
on.

Example — outermost reduction:
```
// P0=1024 is contiguous AND parallel
%0 = input <R0xP0xf16>    // R0=4096, P0=1024
%1 = reduce(R0) ins(%0) -> <P0xf16>
```

If wg_tile[P0] = 1: each WG gets `<4096x1xf16>`. One column. Threads can
only be distributed along R0 (strided) — no coalescing possible.

If wg_tile[P0] = 512: each WG gets `<4096x512xf16>`. Threads can spread
along the 512 contiguous elements — coalesced.

The workgroup tile for P0 directly determines whether coalescing is possible.

## The Coupled Approach

Instead of "tile for parallelism, then coalesce," we need:

**For each operand, determine the minimum per-WG tile along its contiguous
dimension(s) that allows coalesced access. Then tile the parallelizable
groups subject to this constraint.**

### Step 1: Coalescing Requirements

For each tensor in the dispatch, identify which dimension is contiguous
(innermost in memory = last dim for row-major).

Map that tensor dim to its dimension group:
- If the group is a **reduction group** → coalescing happens within the WG
  regardless of WG tile. No constraint on tiling.
- If the group is a **parallel group** → the WG tile for this group must
  be large enough to give threads contiguous elements to work with.

The minimum per-WG size along a contiguous parallel group is:
```
min_coalescing_tile = subgroupSize * vectorWidth
```
This ensures at least one subgroup's worth of coalesced work. Typically
64 * 8 = 512 for f16, 64 * 4 = 256 for f32.

### Step 2: Determine per-group WG tiles with coalescing constraints

```
For each parallelizable group G:
  needs_coalescing = false
  min_tile = 1

  For each tensor T that has a dim in group G:
    If that dim is the contiguous dim of T:
      If T is a "significant" operand (large per-WG):
        needs_coalescing = true
        min_tile = max(min_tile, subgroupSize * vectorWidth)

  wg_tile[G] = max(min_tile, 1)
  // Ensure tile divides group size (or use masking)
```

### Step 3: Compute initial WG count and scale

```
numWGs = product(group.size / wg_tile[group] for each parallel group)

// If too few WGs, scale subgroups
subgroupsPerWG = 1
while numWGs * subgroupsPerWG < targetOccupancy:
    subgroupsPerWG *= 2
    ...

// If too many WGs, coarsen NON-contiguous parallel groups first
// (coarsening the contiguous group reduces coalescing width — last resort)
while numWGs >> targetOccupancy:
    group = best_group_to_coarsen()  // prefer non-contiguous groups
    wg_tile[group] *= 2
    numWGs /= 2
```

## Worked Examples

### Innermost Reduction: P0=1024, R0=4096

```
input <P0xR0xf16>: contiguous dim = R0 (reduction group) → no constraint
output <P0xf16>:   contiguous dim = P0 (parallel group)
  Output is tiny (1 element per WG). Not a significant operand.
  → No coalescing constraint on P0.

wg_tile[P0] = 1. numWGs = 1024.
Coalescing happens along R0 (within WG).
```

This is correct — the contiguous dim is the reduction, so threads naturally
coalesce along R0 within the WG.

### Outermost Reduction: P0=1024, R0=4096

```
input <R0xP0xf16>: contiguous dim is dim 1 = P0 (parallel group!)
  Input is the dominant operand (4096*1024 = 4M elements).
  → NEEDS coalescing along P0.
  min_tile = 64 * 8 = 512 (f16: subgroupSize * 128bits/16bits)

output <P0xf16>: contiguous dim = P0
  Small (1024 elements total). Not driving coalescing.

wg_tile[P0] = 512.  numWGs = 1024 / 512 = 2.
```

Only 2 WGs — not enough. Scale subgroups:
```
subgroupsPerWG = 16: 2 * 16 = 32. Still < 1216.
```

Still not enough. But with 2 WGs each processing 4096*512 = 2M elements
(4MB at f16), memory bandwidth per WG is substantial. 2 WGs might actually
be fine for memory-bound work — the bottleneck is memory, not parallelism.

Alternatively, reduce the coalescing tile:
```
wg_tile[P0] = 128: numWGs = 8. subgroupsPerWG = 16: 8*16 = 128.
  Per-WG: <4096x128xf16> = 512K elements = 1MB.
  128 / 8(vecWidth) = 16 threads along P0. Remaining 48 on R0 or idle.
```

Or:
```
wg_tile[P0] = 64: numWGs = 16. subgroupsPerWG = 16: 16*16 = 256.
  Per-WG: <4096x64xf16> = 256K elements = 512KB.
  64 / 8 = 8 threads along P0.
```

The tradeoff: smaller wg_tile[P0] → more WGs but fewer threads coalesce
along the contiguous dim. The coalescing "quality" degrades but doesn't
disappear — even 8 threads doing 128-bit loads is 8*16 = 128 bytes per
load = 1 cache line on most GPUs.

### Layer Norm: P0=2, P1=32, R0=16384, R1=10

```
input <P0xP1xR1xR0xf16>: contiguous dim = R0 (reduction) → no constraint
output <P0xP1xR1xR0xf32>: contiguous dim = R0 (reduction) → no constraint
mean <P0xP1xf32>: tiny per WG → no constraint

All contiguous dims are reduction groups. No coalescing constraint on P0, P1.

wg_tile[P0] = 1, wg_tile[P1] = 1. numWGs = 64.
subgroupsPerWG = 16. workgroupSize = 1024.
```

Same as before — coalescing is along R0 within the WG, unaffected by
parallel tiling. This is the "easy" case.

### Softmax: P0=512, R0=10240

```
input <P0xR0xf32>: contiguous dim = R0 (reduction) → no constraint
output <P0xR0xf32>: contiguous dim = R0 → no constraint

wg_tile[P0] = 1. numWGs = 512.
subgroupsPerWG = 4. workgroupSize = 256.
```

Again, contiguous dim is reduction. Easy case.

### Matvec: P0=4, P1=4096, R0=4096

```
lhs <P0xR0xf16>: contiguous dim = R0 (reduction) → no constraint on P0, P1
rhs <R0xP1xf16>: contiguous dim = P1 (parallel!)
  rhs is the dominant operand (4096*4096 = 16M elements).
  → NEEDS coalescing along P1.
  min_tile = 64 * 8 = 512

out <P0xP1xf16>: contiguous dim = P1
  Small (4*4096 = 16K). Not driving coalescing.
```

```
wg_tile[P0] = 1, wg_tile[P1] = 512.
numWGs = 4 * (4096/512) = 4 * 8 = 32.
subgroupsPerWG = 16: 32*16 = 512. Close to 1216.

Per-WG:
  lhs: <4x4096xf16> = 32K elements (broadcast across P1 — same for all WGs)
  rhs: <4096x512xf16> = 2M elements = 4MB
  out: <4x512xf16> = 2K elements
```

Threads spread along P1 (512/8 = 64 threads = 1 subgroup). The other 15
subgroups spread along R0 for cooperative reduction.

Compare with fully tiled P1 (wg_tile=1):
```
numWGs = 4*4096 = 16384
Per-WG rhs: <4096x1xf16> = 4096 elements. One column. No coalescing.
```

That would be terrible — each WG reads one column of rhs with strided access.

### Batch Norm: P0=2, P1=32, R0=16384, R1=10
(same as layer norm if iteration space matches)

Contiguous dim = R0 (reduction). No coalescing constraint.

### Batch Norm (NHWC): P_C=256, R_N=64, R_H=56, R_W=56

```
input <R_NxP_CxR_HxR_Wxf16>: contiguous dim = R_W (reduction) → no constraint
output <P_Cxf16>: tiny

Contiguous dim is reduction. No coalescing constraint on P_C.

wg_tile[P_C] = 1. numWGs = 256.
subgroupsPerWG = 4: 256*4 = 1024 ≈ targetOccupancy.
workgroupSize = 256.
```

## Summary

The coupling between tiling and coalescing is:

1. **If the contiguous dim of a significant operand is in a reduction group**:
   No constraint. Coalescing happens within the WG naturally. This is the
   most common case (innermost reduction, layer norm, softmax, batch norm).

2. **If the contiguous dim of a significant operand is in a parallel group**:
   The WG tile for that group must be >= `subgroupSize * vectorWidth` to
   allow coalesced access. This reduces the number of WGs, compensated by
   more subgroups per WG. Cases: outermost reduction, matvec.

The algorithm:
```
for each parallel group G:
  if any significant operand has its contiguous dim in G:
    wg_tile[G] = max(wg_tile[G], subgroupSize * vectorWidth)
  else:
    wg_tile[G] = 1

numWGs = product(sizes / tiles)
scale subgroups to fill GPU
if too many WGs: coarsen non-contiguous groups first
```

This naturally handles both "easy" cases (contiguous=reduction) and "hard"
cases (contiguous=parallel) with a single unified algorithm.
