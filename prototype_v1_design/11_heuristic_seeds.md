# Heuristic Seeds

## Motivation

The workgroup tiling algorithm has several decision points that depend on
heuristic thresholds. Hardcoding these makes it difficult to iterate when
benchmarking. Instead, we collect all tunable parameters into a single
`HeuristicSeeds` struct that can be:

1. Set to sensible defaults
2. Overridden via command-line flags for experimentation
3. Eventually tuned per-target or per-operation-class

## The Struct

```cpp
/// Tunable parameters controlling the workgroup tiling heuristic.
/// All values have sensible defaults. Override via flags for benchmarking.
struct HeuristicSeeds {
  //===--- Coalescing ---===//

  /// Minimum contiguous elements for coalesced access. Threads will be
  /// distributed to cover at least this many contiguous elements per WG.
  /// Default: subgroupSize * vectorWidth (computed from target + element type).
  /// Set lower to trade coalescing quality for more parallelism.
  /// Set higher to ensure wider coalesced accesses (e.g., 2x for prefetching).
  int64_t coalescingTargetElements = 0; // 0 = auto (subgroupSize * vecWidth)

  /// Minimum per-WG operand size (bytes) for the operand to be considered
  /// "significant" for coalescing. Operands smaller than this are assumed
  /// to fit in cache and don't constrain tiling.
  int64_t significantOperandThreshold = 1024; // 1KB

  //===--- Workgroup Count ---===//

  /// Target number of active subgroups across the GPU. The heuristic tries
  /// to ensure numWGs * subgroupsPerWG >= this value.
  /// Default: numWGPs * numSIMDs (from target), or this override.
  int64_t targetOccupancy = 0; // 0 = auto (from target hardware)

  /// Maximum ratio of numWGs to targetOccupancy before we coarsen tiles.
  /// E.g., 4.0 means we start coarsening when numWGs > 4x the target.
  /// Lower = more aggressive coarsening (fewer, larger WGs).
  /// Higher = more WGs (potentially better latency hiding but more overhead).
  float coarsenThreshold = 4.0f;

  //===--- Subgroup Scaling ---===//

  /// Maximum subgroups per workgroup. Limits how much we scale the WG size
  /// when parallelism is insufficient. Capped by maxWorkgroupSize/subgroupSize.
  int64_t maxSubgroupsPerWG = 0; // 0 = auto (maxWorkgroupSize / subgroupSize)

  //===--- Conflict Resolution ---===//

  /// When multiple operands want coalescing along different dimensions,
  /// operands with more than this ratio of total bytes "win" the conflict.
  /// E.g., if input is 100x larger than output, input coalescing wins.
  /// Set to 1.0 to always resolve in favor of the larger operand.
  float coalescingConflictRatio = 1.0f;

  /// Whether to emit a diagnostic when coalescing constraints conflict.
  bool emitCoalescingConflictDiagnostic = true;

  //===--- Per-Thread Work ---===//

  /// Target number of serial iterations per thread along reduction dims.
  /// Higher = more work per thread (better amortization, more register
  /// pressure). Lower = more threads cooperating on reduction.
  int64_t targetBatchPerThread = 8;
};
```

## How Seeds Flow Through the Algorithm

```
Phase A (Coalescing):
  target = seeds.coalescingTargetElements ?: subgroupSize * vectorWidth
  threshold = seeds.significantOperandThreshold

Phase B (Interactions):
  conflict_ratio = seeds.coalescingConflictRatio

Phase C (Tiling):
  targetOccupancy = seeds.targetOccupancy ?: numWGPs * numSIMDs
  coarsenThreshold = seeds.coarsenThreshold
  maxSubgroupsPerWG = seeds.maxSubgroupsPerWG ?: maxWorkgroupSize / subgroupSize
  batchPerThread = seeds.targetBatchPerThread
```

## Usage in Code

```cpp
LogicalResult setMemoryBoundConfig(TargetAttr target,
                                    FunctionOpInterface entryPoint,
                                    LinalgOp op) {
  // Build seeds from target hardware info.
  HeuristicSeeds seeds;
  seeds.targetOccupancy = target.getChip().getWgpCount()
                          * target.getWgp().getSimdsPerWgp().value_or(4);
  seeds.maxSubgroupsPerWG = maxWorkgroupSize / target.getPreferredSubgroupSize();

  auto analysis = analyzeParallelism(entryPoint);
  auto tiling = computeWorkgroupTiling(analysis, seeds);
  ...
}
```

## Benefits

- **Reproducible benchmarking**: Change one seed, re-run, compare.
- **Per-target tuning**: Different GPUs may want different thresholds
  (e.g., larger coalescing target for GPUs with wider memory buses).
- **Gradual refinement**: Start with conservative defaults, tighten based
  on benchmark results.
- **Debuggability**: Print seeds in debug output so it's clear which
  parameters influenced the decision.
