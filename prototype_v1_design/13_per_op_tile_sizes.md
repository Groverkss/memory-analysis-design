# Per-Op Tile Size Computation

## Goal

Given WG tiles from Phase 1, compute per-op minimum tile sizes that obey
coalescing constraints. If tiles end up too large, that's Phase 2's problem.
This algorithm's only job: obey coalescing.

## Output

An internal mapping — no lowering_config changes yet:

```
DenseMap<LinalgOp, SmallVector<int64_t>> minTileSizes;
// minTileSizes[op][iterDim] = minimum tile size for that dim (power of 2)
```

For parallel dims: tile comes from WG tiling (dimension group), shared across
ops. For reduction dims: tile is per-op (different ops can tile the same
reduction group differently).

## Phase 1 fix: per-operand coalescing target

Current Phase 1 uses `elementBits = max(all tensors)`, giving one global
coalescingTarget. Fix: each operand has its own target:

```
coalescingTarget(operand) = subgroupSize * (maxLoadBits / operand.elementBits)
```

f16 → 64 * 8 = 512.  f32 → 64 * 4 = 256.

## Algorithm

### Forward pass (topological order)

For each op:

1. Initialize tile sizes from dimension groups:
   - Parallel dims: tile = wg_tile[group]
   - Reduction dims: tile = group.resolvedSize

2. For each coalesced operand of this op:
   a. Compute coalescingTarget for this operand's element type
   b. Walk operand dims from innermost to outermost
   c. Map each operand dim → iteration dim (via indexing map)
   d. Accumulate product of dim tiles. If the accumulated product reaches
      coalescingTarget, stop.
   e. If a dim's current tile is too small, increase it to the next power of
      2 that helps satisfy the target. Cap at dim's resolved size.

3. Take max across all coalesced operands for each iteration dim.

4. The op's output tile sizes are determined by these iteration dim tiles.
   These become constraints for downstream ops.

### Backward pass (reverse topological order)

For each op:

1. Check coalesced outputs (operands written to global memory).
2. If the output's coalescing requires larger tiles than the forward pass
   set, increase the relevant iteration dim tiles.
3. Propagate back: if increasing a parallel dim tile, update the dimension
   group's WG tile (affects all ops). If increasing a reduction dim tile,
   update only this op.

### Constraint

All tile sizes must be powers of 2.

## Example: innermost reduction 1024×4096 f16

Groups: P0=1024 (parallel), R0=4096 (reduction)
Phase 1 WG tile: P0=1 (coalescing satisfied by R0)

Reduce op: iteration [d0→P0, d1→R0]
- Coalesced input <P0×R0 f16>: innermost R0
  - coalescingTarget(f16) = 512
  - R0 tile = 4096 >= 512 ✓
- minTileSizes = [P0=1, R0=4096]

Store <P0 f16>: innermost P0
- coalescingTarget(f16) = 512
- P0 tile = 1 < 512
- Backward pass: increase P0 WG tile to 512? But P0=1024 total,
  and 512 WGs of size 512 means 2 WGs — reasonable.

## Example: fused layernorm 2×32×10×16384 f16

Groups: P0=2, P1=32, R0=10, R1=16384
Phase 1 WG tile: P0=2, P1=32

Reduce op:
- Coalesced input: innermost R1=16384, target=512 ✓
- minTileSizes = [2, 32, 10, 16384]

Elementwise broadcast op:
- Coalesced output: innermost R1=16384, target=512 ✓
- minTileSizes = [2, 32, 10, 16384]

## Example: transposed output f16

Input <P0×R0 f16>, reduce(R0) → <P0>, store transposed <P0 f16>
Groups: P0=1024 (parallel), R0=4096 (reduction)
Phase 1 WG tile: P0=1

Forward: reduce op, coalesced input innermost R0=4096 ≥ 512 ✓
Backward: store <P0>, innermost P0, target=512. P0 tile=1 < 512.
→ Increase P0 WG tile to 512.

Now per-WG work: P0=512, R0=4096 → 2M elements per WG. Big, but
that's Phase 2's problem.

## Example: mixed precision reduce f16 → f32

Input <P0×R0 f16>, Output <P0 f32>
coalescingTarget(f16) = 512, coalescingTarget(f32) = 256

Forward: coalesced input R0 innermost, R0=4096 ≥ 512 ✓
Backward: store <P0 f32>, target=256. P0 tile needs ≥ 256.
→ P0 WG tile = 256.
