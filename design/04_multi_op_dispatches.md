# Design Addendum: Multi-Op Dispatches

## The Reality

Memory-bound dispatches are almost never a single op. A typical layer norm
dispatch contains:

```
Op 1: extf (elementwise)     — f16→f32 upcast
       (d0,d1,d2,d3) → (d0,d1,d2,d3)    tensor<2x32x10x16384>

Op 2: fill                   — zero init accumulator
       tensor<2x32>

Op 3: sum reduction (mean)   — reduce d2,d3
       ins: (d0,d1,d2,d3) → (d0,d1,d2,d3)   tensor<2x32x10x16384>
       outs: (d0,d1,d2,d3) → (d0,d1)         tensor<2x32>

Op 4: divf (elementwise)     — divide by count
       (d0,d1) → (d0,d1)                     tensor<2x32>

Op 5: variance reduction     — reduce d2,d3 with mean broadcast
       ins: (d0,d1,d2,d3) → (d0,d1,d2,d3)   tensor<2x32x10x16384>
            (d0,d1,d2,d3) → (d0,d1)          tensor<2x32> (mean, broadcast)
       outs: (d0,d1,d2,d3) → (d0,d1)         tensor<2x32>

Op 6: normalization (eltwise) — apply normalization
       ins: (d0,d1,d2,d3) → (d0,d1,d2,d3)   tensor<2x32x10x16384> (input)
            (d0,d1,d2,d3) → (d0,d1)          tensor<2x32> (mean, broadcast)
            (d0,d1,d2,d3) → (d0,d1)          tensor<2x32> (var, broadcast)
       outs: (d0,d1,d2,d3) → (d0,d1,d2,d3)  tensor<2x32x10x16384>
```

Key observations:
- d0, d1 are parallel across ALL ops
- d2, d3 are reduction dims (for the reduction ops) or parallel (for the eltwise ops)
- The reductions produce small tensors (2×32) that get BROADCAST back into
  the larger elementwise ops
- The large tensor (2×32×10×16384) is read by multiple ops

## How This Affects the Design

### The parallelism analysis must consider ALL ops

The "fully parallel" dimensions are those that are parallel in EVERY op.
In the layer norm example:
- d0: parallel in all ops ✓
- d1: parallel in all ops ✓
- d2: parallel in ops 1,6 but REDUCTION in ops 3,5 ✗
- d3: parallel in ops 1,6 but REDUCTION in ops 3,5 ✗

So d0, d1 are the shared parallel dimensions → these get workgroup tiling.
d2, d3 are "mixed" → handled within the workgroup.

### Each op gets its own lowering_config

Looking at the test output, the current code attaches configs to:
- Op 3 (mean reduction): has reduction + expand_dims + basis
- Op 5 (variance reduction): same structure
- Op 6 (normalization eltwise): has serial + basis (no reduction)

The elementwise ops that can be fused into the reductions (op 1, op 4) do
NOT get configs — they get fused into the reduction's tile loop.

### The dispatch has a DAG structure

```
input (2x32x10x16384)
  ↓
extf (fused into reduction)
  ↓
mean_reduction → mean (2x32)
  ↓              ↓ (broadcast)
  ↓         variance_reduction → var (2x32)
  ↓              ↓ (broadcast)        ↓ (broadcast)
  ↓         normalization (2x32x10x16384)
  ↓              ↓
  ↓           output
```

The reductions consume the full tensor and produce small results. Then the
small results are broadcast back into a full-size elementwise op.

### What the config must capture per-op

**For reduction ops (mean, variance)**:
- workgroup tile: [1, 1, 0, 0] (d0,d1 tiled; d2,d3 are reduction)
- partial_reduction: [0, 0, 1, 1024] (chunk size for reduction loop)
- thread: vectorized loads along d3
- lane_basis/subgroup_basis: threads distributed along d3 (innermost)

**For the broadcast+elementwise op (normalization)**:
- workgroup tile: same [1, 1, 0, 0] — must match the reductions
- serial: [0, 0, 1, 8192] (processes full d2,d3 serially in chunks)
- thread: vectorized loads along d3
- lane_basis/subgroup_basis: threads distributed along d3

The key constraint: **all ops in the dispatch share the same workgroup tile
and the same workgroup size**. They must agree on which dimensions are
workgroup-level.

### The broadcast inputs are tiny per-WG

After workgroup tiling (d0=1, d1=1):
- mean per WG: tensor<1x1> = scalar → trivially available, no coalescing
- variance per WG: tensor<1x1> = scalar → same
- input per WG: tensor<1×1×10×16384> = 160K elements → this needs coalescing

The parallelism analysis + workgroup tiling naturally reveals that the
broadcast operands (mean, var) are tiny. No need to optimize their access.

## Revised Design: Multi-Op Aware

### Phase 1: Dispatch-level Parallelism Analysis

Analyze ALL ops in the dispatch together:

```
shared_parallel_dims = intersection of parallel dims across all ops

For each shared parallel dim d:
  For each op:
    For each operand of that op:
      Does d appear in the operand's indexing map?
      → If yes: parallelizing d shrinks this operand
      → If no: this operand is broadcast across d

total_parallelism = product of shared_parallel_dim sizes
```

This is essentially what the current code does with `sharedParallelSet`.

Comment: You can't just take an intersection of the parallel dims like this.
It's not that easy. The iteration space of each op is per op. A dimension in one
iteration space has no corelation to anothers iteration space. This is what the
old analysis does not handle. We have to handle this.

### Phase 2: Workgroup Tiling

Tile the shared parallel dimensions. All ops get the same workgroup tiles.

After tiling, compute per-WG operand sizes for each op:
```
For each op:
  For each operand:
    per_wg_size = product of dims that appear in indexing map / parallel tiles
```

Operands that are broadcast across all parallelized dims stay full-size but
are shared across WGs (L2 cacheable). Operands that shrink become the ones
that need coalescing analysis.

### Phase 3: Subgroup Scaling

Based on total parallelism vs GPU capacity. Same as before.

### Phase 4: Per-Op Thread Layout

Each op gets its own lowering_config, but they share:
- workgroup tile sizes
- workgroup size (number of threads)
- thread distribution pattern (lane_basis)

Wait — do they actually share lane_basis? Looking at the test:

```
// Mean reduction:
lane_basis = [[1, 1, 1, 64, 1], [0, 1, 2, 3, 4]]  (64 threads on d3)
thread = [0, 0, 1, 1, 8]

// Normalization elementwise:
lane_basis = [[1, 1, 1, 64], [0, 1, 2, 3]]  (64 threads on d3)
thread = [0, 0, 0, 8]
```

They have the same EFFECTIVE thread distribution (64 threads on the
innermost dim) but different shapes because the reduction ops have an
extra dimension from expand_dims.

So the basis IS shared conceptually, but the per-op configs encode it
relative to each op's own iteration space.

### Phase 4 detail: Which ops get configs?

Not every op needs a lowering_config. The rules:

1. **Reduction ops**: always get a config (they drive the tiling)
2. **Elementwise ops that can fuse into a reduction's tile loop**: NO config
   (they're fused; the reduction's tiling governs them)
3. **Elementwise ops with broadcast inputs or non-identity output maps**:
   GET a config (they need their own layout anchors because they have
   different dimension structure)
4. **Fill ops**: no config (trivially handled)

This is what `shouldAttachLoweringConfig` in the current code decides.

## Example: Softmax

```
Op 1: max reduction    — (d0,d1) → (d0,d1) ins, (d0,d1) → (d0) out
Op 2: exp + broadcast  — (d0,d1) → (d0) ins (max, broadcast), (d0,d1) → (d0,d1) out
Op 3: sum reduction    — (d0,d1) → (d0,d1) ins, (d0,d1) → (d0) out
Op 4: div + broadcast  — (d0,d1) → (d0) ins (sum, broadcast), (d0,d1) → (d0,d1) out
```

Shared parallel: d0.
d1: reduction in ops 1,3; parallel in ops 2,4.

Workgroup tile: d0=1 (one row per WG).
Per-WG: input row = N elements.

After the max reduction (small result: scalar per row), it's broadcast back
for exp. After sum reduction, broadcast back for div.

All ops share: 64 threads along d1. Reductions use shuffles. Elementwise
ops just have each thread process its portion independently.

The reduction ops and broadcast-elementwise ops get separate configs,
but all with 64 threads on d1 and vecWidth=8 on d1.

## Example: Batch Norm (multi-op)

```
Op 1: sum reduction    — reduce N,H,W keeping C → mean
Op 2: divf             — mean / count
Op 3: variance reduce  — reduce N,H,W keeping C → var (with mean broadcast)
Op 4: normalization    — (input - mean) * rsqrt(var + eps)
```

Shared parallel: C.
N, H, W: reduction in ops 1,3; parallel in op 4.

Workgroup tile: C=1 (one channel per WG).
Per-WG: input = N×1×H×W = 64×56×56 = 200K elements.

Thread layout: 64 threads distributed across H,W for coalesced access
on the per-WG input tile. The reductions use shuffle on the distributed
spatial dims. The normalization writes back with the same thread layout.

## What This Changes in the Algorithm

The algorithm from v3 is mostly correct, but needs these refinements:

1. **Phase 1 operates on the full dispatch**, not a single op. Find shared
   parallel dims across all ops.

2. **Phase 2 workgroup tiling** is dispatch-level. All ops share the tiles.

3. **Phase 4 per-op configs** are set individually, but with a shared thread
   distribution. Each op's config encodes the basis relative to its own
   iteration space (which may differ due to expand_dims).

4. **Broadcast operands are identified naturally**: after workgroup tiling,
   operands that don't shrink are broadcast. They don't need coalescing
   optimization — they're either small (scalars/vectors after reduction)
   or shared (L2 cacheable).

5. **The "dominant operand" for coalescing** emerges from per-WG analysis:
   it's the largest per-WG operand tile, which is typically the full-rank
   input tensor that feeds the reductions.

6. **Elementwise ops between/after reductions** share the same thread layout
   but may process a different tile shape (serial over reduction dims
   instead of reducing). Their configs use "serial" instead of
   "partial_reduction".
