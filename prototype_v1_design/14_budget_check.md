# Budget Check: Drop Coalescing Constraints When Per-WG Work Is Too Large

## Problem

Coalescing constraints can force large WG tiles on parallel dims, making each
WG do an unreasonable amount of work. Example: transposed output with small
parallel dims (P0=32, P1=2) forces the WG to process the entire dispatch in
a single WG of 64 threads, wasting GPU occupancy.

## Overall Workflow

```
constraints = buildCoalescingConstraints(all tensors)
droppedTensors = {}
loop:
  wgTiles = computeWGTiles(constraints, droppedTensors)
  perOpTiles = computePerOpTiles(wgTiles)
  overBudgetOp = findWorstOp(perOpTiles, budget)
  if overBudgetOp:
    tensorToDrop = findConstraintToDrop(constraints, overBudgetOp)
    droppedTensors.add(tensorToDrop)
  else:
    break
```

## Per-WG Work Metric

For each compute op:
```
perWGWork = product(tiles[d] for all iteration dims d of this op)
```

This is the total number of scalar iterations the WG runs for this op.

## Budget

```
budget = subgroupSize * vectorWidth * maxIterationsPerThread
```

`vectorWidth` is per-op, using the widest element type (smallest vector
width, most conservative):
```
vectorWidth = maxLoadBits / max(elemBits across op's operands)
```

`maxIterationsPerThread` is a tunable seed. Default: 1024.

Examples (subgroupSize=64, maxLoadBits=128):
- f16 op: budget = 64 * 8 * 1024 = 524K
- f32 op: budget = 64 * 4 * 1024 = 262K
- mixed f16/f32 op: budget = 64 * 4 * 1024 = 262K (conservative)

Comment: I think the budget should be fixed. It should be a single seed. Maybe
a good default is `subgroupSize * maxLoadBits * k`, this basically restricts us
to at most k bitwidth vectorized operations per thread. We can have k as
something like 8 to start with I think.

## Which Constraint to Drop

When an op is over budget, we need to drop the coalescing constraint that
is most responsible for the excessive work. The constraint to drop is the
one whose removal would reduce the over-budget op's per-WG work the most.

Each constraint is associated with a tensor. Dropping a constraint means
we no longer require coalesced access for that tensor.

Scoring: for each active constraint that affects the over-budget op
(i.e., constrains a parallel group in the op's iteration space):
```
score = product(currentTile / tileWithoutThisConstraint) for each group
        the constraint sets
```

The constraint with the highest score gives the biggest work reduction.

To compute `tileWithoutThisConstraint`: the group's tile would be the max
of all OTHER constraints' min tiles for that group (or 1 if no other
constraint constrains it).

## Example: Transposed Output (Example 07)

P0=32, P1=2, R0=16384, R1=10

Constraints:
1. Output `<R1×R0×P1×P0 f32>`: P0>=32, P1>=2
2. Input `<P1×P0×R1×R0 f16>`: satisfied by reduction (no parallel min tiles)

Reduce op tiles: [2, 32, 10, 16384] → perWGWork = 10.5M
Budget (f32): 262K
Over budget by 40x.

Scoring constraint 1:
- P0: currentTile=32, without constraint=1 → factor 32
- P1: currentTile=2, without constraint=1 → factor 2
- score = 32 * 2 = 64

Drop constraint 1. New tiles: [1, 1, 10, 16384] = 163K < 262K. Done.
Result: 64 WGs, no output coalescing, good occupancy.

## Example: Innermost Reduction (Example 01)

P0=1024, R0=4096

Constraints:
1. Store `<P0 f16>`: P0>=512
2. Input `<P0×R0 f16>`: satisfied by reduction

Reduce op tiles: [512, 4096] → perWGWork = 2M
Budget (f16): 524K
Over budget by ~4x.

Scoring constraint 1:
- P0: currentTile=512, without constraint=1 → factor 512
- score = 512

Drop constraint 1. New tiles: [1, 4096] = 4096 < 524K. Done.
Result: 1024 WGs, no store coalescing, great occupancy.

Hmm — but losing store coalescing for a 1D output might not be ideal.
The budget threshold (maxIterationsPerThread) controls this tradeoff.
With a higher threshold (e.g., 4096), the 2M case would be OK.

Comment: I think losing the coalescing here is good. It's not bad at all!
