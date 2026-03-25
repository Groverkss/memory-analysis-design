# Algorithm Comment (to be placed before computeWorkgroupTiling)

```
//===----------------------------------------------------------------------===//
// Workgroup Tiling Algorithm
//===----------------------------------------------------------------------===//
//
// Decides how to tile parallelizable dimension groups across workgroups.
//
// Input:  Parallelism analysis (dimension groups with resolved sizes),
//         hardware parameters (HeuristicSeeds).
// Output: WG tile for each parallel group, subgroup count per WG.
//
// All sizes used in this algorithm are "resolved sizes" — static sizes when
// known, upper bounds from IntegerRangeAnalysis when dynamic. There is no
// special-casing for dynamic shapes; the algorithm operates on resolved sizes
// uniformly.
//
// The algorithm has 5 phases:
//   Phase 1: Determine coalescing constraints from operand memory layouts.
//   Phase 2: Drop constraints that exceed the per-WG work budget.
//   Phase 3: Analyze dimension interactions (broadcast costs).
//   Phase 4: Set WG count and scale subgroups for occupancy.
//   Phase 5: Generate diagnostic suggestions.
//
// == Phase 1: Coalescing Constraints ==
//
// For each significant tensor in the dispatch, determine minimum WG tile
// sizes for parallel groups to ensure coalesced memory access.
//
// The coalescing target is: subgroupSize * vectorWidth elements. This is the
// number of contiguous elements needed so that a full subgroup can each load
// a vector's worth of contiguous data.
//
// For each tensor, we walk dimensions from innermost (contiguous in memory)
// to outermost:
//   - Reduction dims contribute to coalescing for free (they're always fully
//     within the WG, never tiled across WGs). Their resolved size is
//     subtracted from the remaining coalescing target.
//   - Parallel dims need a minimum WG tile to contribute. The tile is
//     min(group.resolvedSize, remainingTarget). If the parallel dim fully
//     satisfies the remaining target, we stop.
//
// TODO: We currently assume tensors are in row-major order (innermost dim =
// last dim = contiguous). This may not hold for all layouts. The coalescing
// walk should eventually be parameterized by the tensor's actual memory
// layout.
//
// A tensor is "significant" if its non-parallel (reduction) portion is large
// enough that uncoalesced access would hurt (>= significantOperandElements).
// Small tensors (e.g., scalar mean/variance per WG) are skipped.
//
// Multiple tensors may impose constraints on the same parallel groups. We
// take the max across all tensors for each group.
//
// == Phase 2: Budget Check ==
//
// Coalescing constraints from different operands can combine to consume too
// much parallelism. For example, a transposed output might need P0>=256 and
// P1>=64, making each WG process an enormous tile.
//
// We check: does the largest per-WG operand (computed from the current tiles
// and resolved sizes) exceed maxPerWGElements? If so, we drop the least
// valuable coalescing constraints until the budget is met.
//
// Constraints are dropped in order of ascending tensor size — smaller tensors
// benefit less from coalescing and are dropped first. When a constraint is
// dropped, the affected group tiles reset to 1 and are recomputed from the
// remaining constraints.
//
// TODO: The constraint dropping heuristic could be improved. Rather than just
// tensor size, we should account for which constraints give the most "bang
// for the buck" — i.e., dropping a constraint that frees the most parallelism
// relative to the coalescing quality lost.
//
// == Phase 3: Dimension Interactions ==
//
// For each parallel group, compute how tiling it affects operand traffic:
//   - "affected" operands have a dim in the group (tiling shrinks their
//     per-WG size).
//   - "broadcast" operands have NO dim in the group (their per-WG size stays
//     the same regardless of tiling). Each extra WG redundantly loads the
//     full broadcast operand.
//
// The broadcast cost of a group is the total elements of all broadcast
// operands. Higher broadcast cost means tiling this group wastes more
// bandwidth. This informs the coarsening decision in Phase 4.
//
// == Phase 4: Workgroup Count and Scaling ==
//
// 4a. Compute the initial WG count from the parallel group tiles:
//     numWGs = product(group.resolvedSize / wg_tile[group])
//              for all parallel groups.
//
// 4b. If numWGs exceeds coarsenThreshold * targetOccupancy, coarsen tiles
//     to reduce WG count. We iteratively double the tile of the best group:
//       - Prefer groups NOT critical for coalescing.
//       - Among non-critical groups, prefer those with high broadcast cost
//         (coarsening reduces redundant loads the most).
//       - Never coarsen beyond a group's resolved size.
//
// 4c. If numWGs * subgroupsPerWG < targetOccupancy, scale up subgroups per
//     WG. Double subgroupsPerWG until occupancy is met or maxSubgroupsPerWG
//     is reached.
//
// == Phase 5: Suggestions ==
//
// Generate diagnostic suggestions when:
//   - Occupancy is low (numWGs * subgroupsPerWG < targetOccupancy / 2).
//   - A coalescing constraint consumes most of a group's parallelism
//     (tile > resolvedSize / 4), suggesting the dispatch layout is suboptimal.
//
//===----------------------------------------------------------------------===//
```

## Notes on printing

Both parallelism analysis and workgroup tiling share the same pseudo-IR
format using symbolic dimension names. No concrete sizes in the DAG — only
symbols. The legend above the DAG shows resolved sizes:

```
// Dimension symbols:
//   P0 = 32 (parallel)
//   R0 = 4096 (reduction)
//   P1 = ? [resolved: 1048576] (parallel)

%0 = input <P0xR0xf16>
%1 = reduce(R0) ins(%0 : <P0xR0xf16>) -> <P0xf16>
store %1 : <P0xf16>
```

For the workgroup tiling output, the tiling decisions are shown as a
separate section, and the per-WG dispatch reuses the same symbolic DAG.
