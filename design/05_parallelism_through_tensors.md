# Parallelism Analysis: Through Tensors, Not Iteration Spaces

## The Problem

Each linalg op has its own iteration space. `d0` in one op is unrelated to
`d0` in another. You can't intersect iteration space dims across ops.

The connection between ops is the **tensor values** (SSA values) that flow
between them. The parallelism analysis must reason about tensor dimensions,
not iteration space dimensions.

## Tensor Dimensions Are the Invariant

A tensor `tensor<D0 x D1 x ... x Dn>` has fixed dimensions. Different ops
access this tensor via different indexing maps, mapping their own iteration
dims to the tensor's dims.

Example: `tensor<2 x 32 x 10 x 16384 x f32>` in a layer norm dispatch:

```
Op 1 (extf):          iter (d0,d1,d2,d3) → tensor via (d0,d1,d2,d3)
                       All parallel. d0→dim0, d1→dim1, d2→dim2, d3→dim3.

Op 3 (mean reduce):   iter (d0,d1,d2,d3) → tensor via (d0,d1,d2,d3)
                       d0,d1 parallel, d2,d3 reduction.

Op 6 (normalize):     iter (d0,d1,d2,d3) → tensor via (d0,d1,d2,d3)
                       All parallel.
```

Here the indexing maps happen to be identity for all ops, so iteration dims
align. But this isn't guaranteed.

## When Iteration Spaces DON'T Align

Consider a dispatch where the normalization op uses a transposed access:

```
Op A (reduce):  iter (d0,d1) → input via (d0,d1), output via (d0)
                d0=parallel, d1=reduction

Op B (eltwise): iter (d0,d1) → input via (d1,d0), output via (d1,d0)
                d0,d1 both parallel
```

Op A's `d0` maps to input tensor dim 0. Op B's `d0` maps to input tensor
dim 1. They're DIFFERENT tensor dims despite both being called `d0`.

You can't say "d0 is parallel in both ops." You have to say "tensor dim 0
is parallel in op A (maps to A's d0, which is parallel) and parallel in
op B (maps to B's d1, which is parallel)."

## The Right Approach: Tensor-Centric Analysis

### Step 1: Build the tensor graph

The dispatch is a DAG of operations connected by tensor values. Identify:
- **Dispatch inputs**: tensors loaded from global memory (dispatch_tensor_load)
- **Dispatch outputs**: tensors stored to global memory (dispatch_tensor_store)
- **Intermediates**: tensors produced by one op and consumed by another

### Step 2: For each tensor, determine which dims are parallelizable

A dimension `i` of tensor `T` is parallelizable if, in EVERY op that
reads or writes `T`:
- The iteration dim that maps to tensor dim `i` is a parallel iterator

If tensor dim `i` maps to a reduction dim in ANY op, it is NOT
parallelizable for the purpose of workgroup tiling.

```
For tensor T with shape [D0, D1, ..., Dn]:
  For each dim i:
    parallelizable[i] = true
    For each op that accesses T:
      iter_dim = op.indexing_map_for(T).inverse(i)  // which iter dim maps to tensor dim i
      if iter_dim is reduction in op:
        parallelizable[i] = false
```

### Step 3: Propagate through the DAG

Tiling a tensor dimension has cascading effects. If we tile dim 0 of
tensor A, and op X produces A while op Y consumes A:
- Op X must tile its corresponding iteration dim
- Op Y must tile its corresponding iteration dim
- This may affect OTHER tensors that X and Y access

The analysis must propagate: "if I tile this tensor dim, what iteration
dims in each op are affected, and what other tensor dims are consequently
tiled?"

Example:
```
Op X: iter (a,b) → writes T1 via (a,b), reads T0 via (a,b)
Op Y: iter (c,d) → reads T1 via (d,c), writes T2 via (c,d)
```

Tiling T1's dim 0:
- Op X: T1 dim 0 = iter dim a → tile a
  - T0 is accessed via (a,b) → T0 dim 0 is also tiled
- Op Y: T1 dim 0 = iter dim d → tile d
  - T2 is accessed via (c,d) → T2 dim 1 is also tiled

So tiling T1 dim 0 also tiles T0 dim 0 and T2 dim 1. These must ALL
be parallelizable for the tiling to be valid.

### Step 4: Find the parallelizable dimension groups

From the propagation, we get groups of (tensor, dim) pairs that are
linked — tiling one requires tiling all of them. Each group is either
fully parallelizable (all corresponding iter dims are parallel) or not.

The parallelizable groups are candidates for workgroup tiling.

## Simplification for Common Cases

In practice, most memory-bound dispatches have a simple structure:

1. **Same-shape ops with identity maps**: All ops share the same iteration
   space shape and use identity (or near-identity) indexing maps. The
   iteration dims DO align. This covers most reductions, softmax, layer norm
   when not transposed.

2. **Reduction + broadcast pattern**: A reduction produces a smaller tensor
   (dropping some dims), then a broadcast-elementwise consumes it alongside
   the full tensor. The reduced-away dims are NOT in the small tensor, so
   they're trivially not parallelizable (they don't exist in the output).
   The surviving dims in the small tensor correspond to parallel dims.

3. **Transposed access**: Less common but does happen (e.g., NHWC↔NCHW).
   This is where the naive "intersect d0 across ops" breaks.

For cases 1 and 2, we could use a simplified analysis that tracks tensor
dims through the DAG without full propagation. For case 3, we need the
full analysis.

## What This Means for the Config

The workgroup tiling is expressed in terms of the dispatch output tensor's
dimensions (since that's what gets stored). For each op, the workgroup tile
maps to its iteration space through its output indexing map.

The per-op configs translate the dispatch-level tiling into each op's own
iteration space:
- Op A's workgroup tile = dispatch tile projected through A's output map
- Op A's lane_basis = dispatch thread layout projected through A's maps

The translation is mechanical once we know the dispatch-level tiling and
thread layout.

## Concrete Example: Layer Norm (revisited)

Dispatch tensors:
- Input: `tensor<2x32x10x16384xf16>` (global input)
- Output: `tensor<2x32x10x16384xf32>` (global output, normalized)
- Intermediate: `tensor<2x32xf32>` (mean), `tensor<2x32xf32>` (variance)

Output tensor dims: [2, 32, 10, 16384].

Parallelizability of output dims:
- Dim 0 (size 2): In all ops, maps to parallel iter dim → ✓
- Dim 1 (size 32): In all ops, maps to parallel iter dim → ✓
- Dim 2 (size 10): In reduction ops, maps to reduction → ✗
- Dim 3 (size 16384): In reduction ops, maps to reduction → ✗

Propagation: tiling output dim 0 tiles:
- Op 6 iter dim 0 → also tiles its input (full tensor) dim 0, mean dim 0, var dim 0
- Op 5 (variance): mean dim 0 → iter dim 0 (parallel) ✓
- Op 3 (mean): input dim 0 → iter dim 0 (parallel) ✓
- Op 1 (extf): input dim 0 → iter dim 0 (parallel) ✓

All consistent. Output dims 0,1 are parallelizable. Dims 2,3 are not.

Workgroup tile (on output): [1, 1, -, -] → 2×32 = 64 workgroups.

Each op gets its own config with workgroup tiles mapped through its own
indexing maps. Since all ops happen to use identity maps here, the
iteration-space workgroup tiles are [1, 1, 0, 0] for all of them.

## When It Gets Interesting

Consider a hypothetical dispatch where one op transposes:

```
Op A (reduce):   iter (m,n,k) → input via (m,n,k), output via (m,n)
                 k=reduction

Op B (eltwise):  iter (p,q,r) → reduced via (p,q), other via (r,q,p), out via (r,q,p)
                 all parallel
```

Output tensor of the dispatch: via Op B's out, shape follows (r,q,p).

Op B reads `reduced` tensor via (p,q). So:
- reduced dim 0 ↔ Op B iter dim p
- reduced dim 1 ↔ Op B iter dim q

Op A writes `reduced` via (m,n). So:
- reduced dim 0 ↔ Op A iter dim m (parallel)
- reduced dim 1 ↔ Op A iter dim n (parallel)

Op B reads `other` via (r,q,p). `other` has shape corresponding to
dims (r,q,p) in Op B. If `other` is a dispatch input, its dims are
independent.

Tiling output dim 0 (Op B's r):
- Op B iter r → other dim 0 gets tiled, but r doesn't appear in
  reduced's map (p,q) → reduced is unaffected
- Only `other` and `out` are affected

Tiling output dim 2 (Op B's p):
- Op B iter p → reduced dim 0 gets tiled
- → Op A iter m gets tiled (m is parallel) ✓
- → Also tiles Op A's input dim 0

So the parallelizable groups are:
- Group 1: {output dim 0, other dim 0} — tiles Op B's r only
- Group 2: {output dim 1, other dim 1, reduced dim 1} — tiles Op B's q, Op A's n
- Group 3: {output dim 2, other dim 2, reduced dim 0, input dim 0} — tiles Op B's p, Op A's m

Each group can be independently tiled if all corresponding iter dims are
parallel.

## Summary

The parallelism analysis must:
1. Reason about **tensor dimensions**, not iteration space dims
2. **Propagate** tiling decisions through the op DAG via indexing maps
3. Find **groups of linked tensor dims** that must be tiled together
4. Check that each group is fully parallelizable (no reduction anywhere)
5. Select which groups to tile for workgroup distribution

This is more complex than "intersect parallel dims" but it's the correct
foundation for handling arbitrary multi-op dispatches.
