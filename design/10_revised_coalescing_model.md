# Revised Coalescing Model

## Key Correction: Reduction Dims Are Part of the Contiguous Region

The contiguous region in memory includes ALL dimensions — parallel AND
reduction. A reduction dim is contiguous in memory just like any other dim.
The only difference is that reduction dims can't be tiled across workgroups.

Previous (wrong) model:
```
Walk from innermost dim, stop at first reduction dim.
```

Correct model:
```
Walk from innermost dim. The ENTIRE tensor is contiguous in row-major layout.
The coalescing region spans all dims. But only the PARALLEL dims within
the region can be tiled to provide WG-level coalescing.
```

### What This Changes

For `<P0 x P1 x R1 x R0 x f16>` (layer norm input):

The contiguous region is: R0 (innermost), R1, P1, P0. All contiguous.

For coalescing, threads will be spread from innermost outward within the
WG tile. R0 and R1 are always fully within the WG (they're reduction —
not tiled across WGs). So:

```
elements_from_reduction = R0 * R1 = 16384 * 10 = 163840
coalescing_target = subgroupSize * vectorWidth = 64 * 8 = 512
```

Since `elements_from_reduction (163840) >> coalescing_target (512)`, the
reduction dims alone provide MORE than enough contiguous elements for
coalescing. No constraint on P0 or P1.

This matches the conclusion before, but the REASONING is different. We don't
stop at the first reduction dim — we recognize that reduction dims contribute
to the contiguous region and can satisfy the coalescing target by themselves.

### When Reduction Dims Aren't Enough

Consider `<P0 x P1 x R0 x f16>` where R0 = 32:

```
elements_from_reduction = R0 = 32
coalescing_target = 512
remaining = 512 / 32 = 16
```

R0 alone isn't enough. We need P1 to provide more contiguous elements:
```
wg_tile[P1] >= 16
```

And if P1 is also small (say P1 = 8):
```
remaining after P1 = 16 / 8 = 2
wg_tile[P0] >= 2
```

So the general algorithm walks from innermost dim outward, "consuming"
contiguous elements:
- Reduction dims: fully consumed (they're always in the WG). Subtract
  from target.
- Parallel dims: set wg_tile to consume remaining target.

### Revised Algorithm

```
For tensor T, walking dims from innermost to outermost:
  remaining_target = subgroupSize * vectorWidth

  for dim in [T.rank-1, T.rank-2, ...]:
    group = findGroup(T, dim)
    dim_size = T.dimSize(dim)

    if group is reduction:
      // This dim is fully within the WG. It contributes to coalescing
      // without any constraint on workgroup tiling.
      remaining_target = ceil(remaining_target / dim_size)
      if remaining_target <= 1:
        break  // coalescing satisfied, no parallel constraints needed

    if group is parallel:
      // This dim needs a minimum WG tile to contribute to coalescing.
      needed = min(dim_size, remaining_target)
      wg_tile[group] = max(wg_tile[group], needed)
      remaining_target = ceil(remaining_target / needed)
      if remaining_target <= 1:
        break
```

---

## Conflicting Input/Output Coalescing

### The Problem

Consider:
```
input:  <P0 x R0 x P1>    // P1 is innermost (contiguous)
output: <P1 x P0>          // P0 is innermost (contiguous)
reduce: R0
```

Input wants P1 coalesced (innermost). Output wants P0 coalesced (innermost).
They disagree on which parallel group should get the most threads.

### Analysis

Input coalescing: walk from P1 (innermost):
```
P1 is parallel: wg_tile[P1] >= 512 (assuming target=512)
```

Output coalescing: walk from P0 (innermost):
```
P0 is parallel: wg_tile[P0] >= 512
```

Both constraints together: wg_tile[P0] >= 512, wg_tile[P1] >= 512.
numWGs = (1024/512) * (1024/512) = 2 * 2 = 4.

Per-WG:
```
input:  <512 x 4096 x 512> = 1G elements. Huge!
output: <512 x 512>        = 256K elements
```

That per-WG input is enormous — keeping 512 along BOTH P0 and P1 makes
the WG tile massive.

### The Tradeoff

We can't satisfy both coalescing constraints without a huge per-WG tile.
The options are:

**Option A: Prioritize input coalescing (larger operand).**
```
wg_tile[P1] = 512, wg_tile[P0] = 1
numWGs = 1024 * 2 = 2048
Per-WG input:  <1 x 4096 x 512> = 2M elements. Manageable.
Per-WG output: <512 x 1>        = 512 elements. Small, strided writes.
```
Output writes are uncoalesced (P0=1, but P0 is output's contiguous dim).
But output is small per WG (512 elements = 1KB). Cache handles it.

**Option B: Prioritize output coalescing.**
```
wg_tile[P0] = 512, wg_tile[P1] = 1
numWGs = 2 * 1024 = 2048
Per-WG input:  <512 x 4096 x 1> = 2M elements. Strided reads!
Per-WG output: <1 x 512>        = 512 elements. Coalesced.
```
Input reads are uncoalesced (P1=1, but P1 is input's contiguous dim).
Input is 4MB per WG — strided reads on 4MB is very bad.

**Option A is clearly better**: sacrifice output coalescing (tiny operand)
to keep input coalescing (dominant operand).

Comment: I think the answer should be more "prioritize larger operand
coalescing" rather than "always prioritize input". But in practice, inputs are
usually larger.

### Decision Rule

When input and output coalescing constraints conflict:

```
For each conflicting parallel group G:
  input_bytes_affected = sum of per-WG sizes of inputs that need G for coalescing
  output_bytes_affected = sum of per-WG sizes of outputs that need G for coalescing

Prioritize the side with larger total bytes.
```

In practice: inputs almost always dominate for reductions (input is
read across all reduction iterations; output is written once). So
**input coalescing wins by default**.

For the pathological case where output is also large (e.g., the normalize
op in layer norm writes a full-size tensor), both input and output usually
have the SAME contiguous dim, so there's no conflict.

Comment: But there could be a conflict there. We can hope there isn't but there
can be.

### When There IS a Real Conflict

A real conflict happens when:
1. Input contiguous dim ≠ output contiguous dim
2. Both operands are large

This typically indicates a **badly fused dispatch** — the producer and
consumer have incompatible memory layouts. The right fix is at the
dispatch level (transpose, or split into separate dispatches).

In the compiler, we should:
1. Detect this conflict
2. Prioritize the larger operand
3. **Emit a diagnostic/remark** suggesting the user/dispatch-formation
   could improve things by ensuring compatible layouts.

---

## Updated Coalescing Algorithm

```
function computeCoalescingConstraints(tensors, groups):
  wg_tile_min = {G: 1 for G in parallel_groups}
  conflicts = []

  for each significant tensor T:
    remaining = subgroupSize * vectorWidth(T)
    for dim in reversed(range(T.rank)):  // innermost first
      group = findGroup(T, dim)
      dim_size = group.size  // (or T.dimSize(dim) if static)

      if group is reduction:
        remaining = ceil(remaining / dim_size)
      else:  // parallel
        needed = min(dim_size, remaining)
        wg_tile_min[group] = max(wg_tile_min[group], needed)
        remaining = ceil(remaining / needed)

      if remaining <= 1:
        break

  // Check for conflicts: same group pulled in different directions
  // by different tensors. If so, prioritize by operand size.
  // (Implementation detail: track which tensor set each constraint.)

  return wg_tile_min
```

---

## Revised Full Example: Layer Norm

```
// P0=2, P1=32, R0=16384, R1=10
input  <P0 x P1 x R1 x R0 x f16>   // 160K elements per WG
output <P0 x P1 x R1 x R0 x f32>   // 160K elements per WG
mean   <P0 x P1 x f32>              // scalar per WG
var    <P0 x P1 x f32>              // scalar per WG
```

**Input coalescing** (innermost = R0):
```
remaining = 512
R0 = 16384 (reduction): remaining = ceil(512/16384) = 1. Done!
```
No parallel constraint needed. R0 alone exceeds the coalescing target.

**Output coalescing** (innermost = R0):
Same — R0 alone satisfies it.

Result: wg_tile = [P0=1, P1=1]. No coalescing constraint on parallel groups.

This is the same conclusion as before, but now we correctly account for
reduction dims as part of the contiguous region rather than stopping at
them.

---

## Revised Full Example: Small-R0 Layer Norm Variant

```
// P0=2, P1=32, R0=32, R1=10
input <P0 x P1 x R1 x R0 x f16>
```

**Input coalescing** (innermost = R0):
```
remaining = 512
R0 = 32 (reduction): remaining = ceil(512/32) = 16
R1 = 10 (reduction): remaining = ceil(16/10) = 2
P1 = 32 (parallel):  wg_tile[P1] >= 2. remaining = 1. Done!
```

Here R0*R1 = 320 isn't enough for the 512 target, so P1 needs a minimum
tile of 2. This means:

```
wg_tile = [P0=1, P1=2]
numWGs = 2 * 16 = 32
```

The constraint comes from the COMBINATION of reduction and parallel dims
in the contiguous region. Stopping at R0 would have missed this.

---

## Conflicting Coalescing Example

```
// P0=1024, R0=4096, P1=1024
input:  <P0 x R0 x P1 x f16>    // P1 innermost
output: <P1 x P0 x f16>          // P0 innermost
reduce: R0
```

**Input coalescing** (innermost = P1):
```
remaining = 512
P1 (parallel): wg_tile[P1] >= 512. remaining = 1. Done.
```

**Output coalescing** (innermost = P0):
```
remaining = 512
P0 (parallel): wg_tile[P0] >= 512. remaining = 1. Done.
```

Both constraints: wg_tile[P0] >= 512, wg_tile[P1] >= 512.
Per-WG input = 512 * 4096 * 512 = 1G elements. Way too large.

**Conflict resolution**: Compare operand sizes.
```
input_size  = P0 * R0 * P1 = 1024 * 4096 * 1024 = 4G elements
output_size = P1 * P0       = 1024 * 1024          = 1M elements
```

Input is 4000x larger. **Prioritize input coalescing.**

```
wg_tile = [P0=1, P1=512]
numWGs = 1024 * 2 = 2048
Per-WG input:  <1 x 4096 x 512> = 2M elements ✓
Per-WG output: <512 x 1>        = 512 elements (uncoalesced writes, but tiny)
```

Emit diagnostic: "Output tensor has incompatible layout for coalescing.
Consider transposing the output or restructuring the dispatch."
