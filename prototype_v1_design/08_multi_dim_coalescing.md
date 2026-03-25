# Design: Multi-Dimensional Coalescing Constraints

## The Problem

Coalescing can span multiple contiguous dimensions. The constraint isn't
per-dimension — it's on the **product** of all parallel groups that form
an unbroken contiguous block from the innermost dim outward.

## Contiguous Parallel Region

For a tensor `<D0 x D1 x ... x Dn>` in row-major layout:
- Dn is innermost (contiguous)
- D(n-1) is next: D(n-1)*Dn is contiguous
- etc., all the way to D0

The **contiguous parallel region** is the longest suffix of consecutive
dimensions (starting from innermost) where ALL dimensions belong to
parallelizable groups. As soon as we hit a reduction dim, we stop.

```
For tensor T with dims [D0, D1, D2, D3] (D3 innermost):
  contiguous_parallel_region = []
  for dim in [D3, D2, D1, D0]:  // innermost to outermost
    group = findGroup(T, dim)
    if group.isParallelizable:
      contiguous_parallel_region.append(group)
    else:
      break  // hit a reduction dim, stop
```

The coalescing constraint is:
```
product(wg_tile[g] for g in contiguous_parallel_region) >= subgroupSize * vectorWidth
```

NOT each wg_tile individually.

## Example: Reduce-N in NHWC

```
input: tensor<N x H x W x C>   (C contiguous)
output: tensor<H x W x C>
reduce: N
parallel: H, W, C
```

Dimension groups: R_N (reduction), P_H, P_W, P_C (all parallel).

Input tensor dims (innermost first): C=P_C, W=P_W, H=P_H, N=R_N.
Contiguous parallel region = {P_C, P_W, P_H}.

Constraint: `wg_tile[P_C] * wg_tile[P_W] * wg_tile[P_H] >= 512` (f16).

Possible tilings:
```
a) wg_tile = [P_H=1, P_W=1, P_C=512]: numWGs = H*W*C/512
   → Only coalesces along C. Fine if C >= 512.

b) wg_tile = [P_H=1, P_W=8, P_C=64]: product = 512. numWGs = H*W*C / 512
   → Coalesces across 64 C elements and 8 W elements.
   → Threads: 64/vecWidth along C, then along W.

c) wg_tile = [P_H=4, P_W=8, P_C=16]: product = 512. numWGs = H*W*C / 512
   → Spreads across all three. Good when C is small.
```

The choice depends on dim sizes. If C=256, W=56, H=56:
```
Option a: P_C=256(capped), product=256 < 512. Need more from W.
  → P_C=256, P_W=2: product=512. numWGs = 56*28*1 = 1568.

Option b: P_C=64, P_W=8: product=512. numWGs = 56*7*4 = 1568.

Option c: P_C=16, P_W=8, P_H=4: product=512. numWGs = 14*7*16 = 1568.
```

All give the same numWGs! The product constraint is what matters.

## Example: Batch Norm in NHWC

```
input: tensor<N x H x W x C>   (C contiguous)
output: tensor<C>
reduce: N, H, W
parallel: C only
```

Input tensor dims (innermost first): C=P_C, W=R_W, H=R_H, N=R_N.
Contiguous parallel region = {P_C} (W is reduction, stops the chain).

Constraint: `wg_tile[P_C] >= 512`.

If C=256: wg_tile[P_C] = 256 (capped). Product = 256 < 512.
Can't meet the full constraint. Accept partial coalescing (256 contiguous
elements is still 4 cache lines — not terrible).

numWGs = 1. Need heavy subgroup scaling.

This is inherently low-parallelism — batch norm has only C parallel.

## Example: Batch Norm in NCHW

```
input: tensor<N x C x H x W>   (W contiguous)
output: tensor<C>
reduce: N, H, W
parallel: C
```

Input tensor dims (innermost first): W=R_W, H=R_H, C=P_C, N=R_N.
Contiguous parallel region = {} (innermost dim W is reduction).

No coalescing constraint on parallel groups. Coalescing happens within
the WG along the reduction dims W (and H). This is the easy case.

## How to Fill the Contiguous Parallel Region

When we have multiple parallel groups in the contiguous region, we fill
from innermost outward (matching memory order):

```
remaining = subgroupSize * vectorWidth
for group in contiguous_parallel_region:  // innermost first
  wg_tile[group] = min(group.size, remaining)
  remaining = ceil(remaining / wg_tile[group])
  if remaining <= 1:
    break
```

This ensures:
- The innermost parallel dim gets the most threads (best coalescing)
- Outer parallel dims get remaining threads
- The product meets the target

## Thread Distribution Within the WG

The lane_basis follows the same order. For the contiguous parallel region:
```
vectorWidth applied to innermost parallel dim
threads fill from innermost parallel dim outward
```

For `wg_tile = [P_H=4, P_W=8, P_C=16]` with vectorWidth=8 (f16):
```
threads_on_P_C = 16/8 = 2   (2 threads, each loads 8 elements)
threads_on_P_W = 8/1 = 8    (8 threads, each handles 1 W position)
threads_on_P_H = 4/1 = 4    (4 threads, each handles 1 H position)
total = 2 * 8 * 4 = 64 = subgroupSize ✓
```

The lane_basis would encode this as:
```
counts  = [4, 8, 2]         // P_H, P_W, P_C (outermost to innermost)
mapping = [iter_H, iter_W, iter_C]
```

Thread 0: C=0, W=0, H=0  → address = 0
Thread 1: C=1, W=0, H=0  → address = 8 (vectorWidth elements later)
Thread 2: C=0, W=1, H=0  → address = C_size
Thread 3: C=1, W=1, H=0  → address = C_size + 8
...

Wait, that's not right. For memory coalescing, we want consecutive
threads to access consecutive addresses. C is innermost (stride 1 in
memory). So:

Thread 0 loads C[0:8], Thread 1 loads C[8:16] → contiguous ✓
Thread 2 loads W=1, C[0:8] → address = C_size + 0
Thread 1's last element: C[15], address = 15
Thread 2's first element: C[0] at W=1, address = C_total

If C_total = 16 (the WG tile): Thread 1 ends at address 15, Thread 2
starts at address 16. Contiguous! ✓

Thread 15 (last in first W row): W=0, loads C[...] up to address W-1
Thread 16 (first in second W row): W=1, address = C_total * 1

As long as the tensor dims are contiguous in memory (standard row-major),
spanning across W and H boundaries is also contiguous.

## When the Contiguous Region is Mixed (Parallel + Reduction)

Consider `tensor<P0 x R0 x P1 x P2>` (P2 innermost):
- Contiguous parallel region = {P2, P1} (R0 breaks the chain)
- P0 is parallel but NOT contiguous with P1 (R0 is between them)

Coalescing constraint: `wg_tile[P1] * wg_tile[P2] >= target`.
P0 has no coalescing constraint from this operand.

P0 can be tiled to 1 (max parallelism) without hurting coalescing.

## Algorithm Update

```
For each significant operand T in the dispatch:
  // Find contiguous parallel region (innermost-outward)
  contiguous_groups = []
  for dim in reversed(range(T.rank)):  // innermost first
    group = findGroup(T, dim)
    if group.isParallelizable:
      contiguous_groups.append(group)
    else:
      break

  // Apply coalescing constraint to product of tiles
  target = subgroupSize * vectorWidth
  remaining = target
  for group in contiguous_groups:  // innermost first
    min_tile = min(group.size, remaining)
    wg_tile[group] = max(wg_tile[group], min_tile)
    remaining = ceil(remaining / min_tile)
    if remaining <= 1:
      break

// After processing all operands, wg_tile has the maximum across all
// operands' coalescing requirements for each group.

numWGs = product(group.size / wg_tile[group] for parallel groups)
// Scale subgroups, coarsen non-contiguous groups if needed, etc.
```

## Summary

- Coalescing spans multiple contiguous parallel dimensions
- The constraint is on the **product** of WG tiles, not each individually
- Fill from innermost dim outward (matching memory order)
- Different operands may have different contiguous regions; take the max
- Non-contiguous parallel groups (separated by reduction dims) have no
  coalescing constraint and can be tiled to 1
