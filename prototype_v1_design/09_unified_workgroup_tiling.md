# Unified Design: Workgroup Tiling Algorithm

## Inputs

From the parallelism analysis:
- **Dimension groups**: P0, P1, ... (parallel) and R0, R1, ... (reduction),
  each with a size (static or dynamic).
- **Dispatch DAG**: tensors with symbolic shapes and ops connecting them.

From hardware:
- `subgroupSize` (e.g., 64)
- `maxWorkgroupSize` (e.g., 1024)
- `numWGPs`, `numSIMDs` → `targetOccupancy = numWGPs * numSIMDs`
- `maxLoadBits` (e.g., 128) → `vectorWidth = maxLoadBits / elementBitWidth`

## Outputs

- `wg_tile[G]` for each parallel group G
- `subgroupsPerWG`
- Per-WG operand shapes for each tensor

---

## Phase A: Coalescing Analysis

For each tensor in the dispatch, determine what coalescing constraints
it imposes on the parallel group tiles.

### A.1: Find the contiguous parallel region

For each tensor T, walk dims from innermost to outermost:
```
contiguous_parallel_region(T) = []
for dim in [T.rank-1, T.rank-2, ...]:   // innermost first
  group = findGroup(T, dim)
  if group.isParallelizable:
    contiguous_parallel_region.append(group)
  else:
    break   // reduction dim breaks the chain
```

### A.2: Compute per-tensor coalescing target

Not every tensor needs coalescing. Small operands fit in cache — don't
constrain the tiling for them. We use per-WG estimated size to decide.

But we have a chicken-and-egg problem: per-WG size depends on the tiles
we haven't chosen yet. Solution: compute the "intrinsic" size — the size
of the tensor along NON-parallel dimensions (reduction dims + any parallel
dims NOT in the contiguous region). This is the per-WG size when all
contiguous parallel dims are tiled to 1 (worst case).

```
intrinsic_size(T) = product of T's dim sizes for dims NOT in any
                    contiguous parallel group
                  = product of reduction-dim sizes and non-contiguous
                    parallel-dim sizes
```

If `intrinsic_size(T) * elementBytes` is large (say > 1KB), the tensor is
**significant** and its coalescing constraint matters. Otherwise, it fits
in cache regardless — skip it.

### A.3: Collect coalescing constraints

```
coalescing_constraints = []
for each tensor T:
  region = contiguous_parallel_region(T)
  if region is empty:
    continue  // innermost dim is reduction — no parallel constraint
  if intrinsic_size(T) * elementBytes < threshold:
    continue  // small operand — skip

  coalescing_constraints.append({
    groups: region,         // [G_innermost, G_next, ...]
    target: subgroupSize * vectorWidth(T),
    tensor: T
  })
```

Multiple tensors may impose constraints on the same groups. We take the
union: each group's minimum tile is the max across all constraints.

---

## Phase B: Dimension Interaction Analysis

When we tile a parallel group, it affects some operands but not others.
Understanding this helps us make good tiling choices.

### B.1: Per-group operand impact

For each parallel group G:
```
affected_operands(G)    = tensors that have a dim in group G
unaffected_operands(G)  = tensors that do NOT have any dim in group G
                        = tensors where G is "broadcast"
```

Tiling group G reduces per-WG size for affected operands but does NOT
reduce it for unaffected (broadcast) operands.

### B.2: Broadcast cost

If a tensor T is unaffected by group G, then every WG reads the full T
(or the same slice of T). This creates redundant global memory traffic
if there are many WGs:

```
broadcast_cost(G) = sum of size(T) for T in unaffected_operands(G)
```

More WGs along G → more total broadcast traffic. This is a cost of
parallelism along G.

### B.3: Compute cost ratio

The net benefit of tiling G depends on what we save vs what we waste:

```
saved_per_wg(G) = sum of (size_of_G_dims_in_T) for T in affected_operands(G)
broadcast_cost(G) = sum of size(T) for T in unaffected_operands(G)
```

If `broadcast_cost(G) >> saved_per_wg(G)`, tiling G aggressively wastes
bandwidth on broadcast reads. Better to keep G coarse.

This is particularly relevant for **matvec**:
- Tiling P1 (N): saves rhs per WG, but lhs is broadcast → redundant lhs loads.
- Tiling P0 (M): saves lhs per WG, but rhs is broadcast → redundant rhs loads.

---

## Phase C: Workgroup Tile Selection

### C.1: Set minimum tiles from coalescing

```
wg_tile[G] = 1 for all parallel groups G

for each constraint in coalescing_constraints:
  remaining = constraint.target  // e.g., 512
  for group in constraint.groups:  // innermost first
    needed = min(group.size, remaining)
    wg_tile[group] = max(wg_tile[group], needed)
    remaining = ceil(remaining / needed)
    if remaining <= 1:
      break
```

After this, the coalescing-critical groups have their minimum tiles set.

### C.2: Compute WG count

```
numWGs = product(group.size / wg_tile[group] for each parallel group)
```

### C.3: Scale subgroups if not enough WGs

```
subgroupsPerWG = 1
while numWGs * subgroupsPerWG < targetOccupancy:
  subgroupsPerWG *= 2
  if subgroupsPerWG * subgroupSize > maxWorkgroupSize:
    subgroupsPerWG /= 2
    break
```

### C.4: Coarsen if too many WGs

When `numWGs` is much larger than `targetOccupancy`, we coarsen tiles to
reduce WG count and increase per-WG work.

**Which group to coarsen?** Priority order:
1. Groups NOT in any coalescing constraint (no impact on coalescing)
2. Groups with high broadcast cost (coarsening reduces redundant loads)
3. Larger groups (more headroom to coarsen)

```
while numWGs > coarsenThreshold * targetOccupancy:
  G = pick_best_group_to_coarsen()
  wg_tile[G] *= 2
  numWGs /= 2
```

The `pick_best_group_to_coarsen` function scores each candidate:
```
score(G) = broadcast_cost(G)       // higher = more benefit from coarsening
         + (not_in_coalescing * bonus)  // prefer non-coalescing groups
```

### C.5: Record per-WG operand shapes

```
for each tensor T:
  per_wg_shape(T) = []
  for each dim d of T:
    group = findGroup(T, d)
    if group.isParallelizable:
      per_wg_shape.append(group.size / wg_tile[group])
      // = wg_tile[group] if we're tiling TO that size, not BY it
      // Actually: per_wg_dim = wg_tile[group] (each WG processes this many)
    else:
      per_wg_shape.append(T.dimSize(d))  // full reduction dim
```

---

## Worked Examples

### Example 1: Innermost Reduction (layer norm)

```
// P0=2, P1=32, R0=16384, R1=10
%0 = input <P0xP1xR1xR0xf16>
%4 = elementwise ins(%0) -> <P0xP1xR1xR0xf32>
%5 = reduce(R1, R0) ins(%4) -> <P0xP1xf32>
%6 = elementwise ins(%5) -> <P0xP1xf32>
%7 = reduce(R1, R0) ins(%4, %6) -> <P0xP1xf32>
%8 = elementwise ins(%3, %6, %7) -> <P0xP1xR1xR0xf32>
```

**Phase A (Coalescing)**:

Input `<P0xP1xR1xR0xf16>` (innermost=R0):
  contiguous_parallel_region = [] (R0 is reduction, breaks immediately)
  → No constraint.

Comment: Well what if for coalescing you needed P1, R1, AND R0? You should
consider the entire contiguous coalescing region and then see if something is
parallel inside it. reduction dimensions aren't special for coalescing, they
are just something you can't tile over.

Output `<P0xP1xR1xR0xf32>` (innermost=R0):
  Same — no constraint.

Mean/var `<P0xP1xf32>`: tiny after WG tiling. Skip.

**No coalescing constraints on parallel groups.** All coalescing is within
the WG along R0. Easy case.

**Phase B (Interactions)**:
- Tiling P0: affects all operands (P0 appears everywhere). No broadcasts.
- Tiling P1: affects all operands. No broadcasts.

**Phase C (Tiling)**:
```
wg_tile = [P0=1, P1=1]
numWGs = 2 * 32 = 64
targetOccupancy = 1216
subgroupsPerWG = 16 → workgroupSize = 1024
numWGs * subgroupsPerWG = 64 * 16 = 1024 ≈ targetOccupancy ✓
```

**Per-WG shapes**:
```
input:  <1 x 1 x 10 x 16384 x f16>  = 160K elements
output: <1 x 1 x 10 x 16384 x f32>  = 160K elements
mean:   <1 x 1 x f32>                = scalar
var:    <1 x 1 x f32>                = scalar
```

**Config**:
```
workgroup = [1, 1, 0, 0]
subgroupsPerWG = 16
// Threads coalesce along R0 (16384 contiguous) within the WG.
```

---

### Example 2: Outermost Reduction

```
// P0=1024, R0=4096
%0 = input <R0xP0xf16>
%1 = reduce(R0) ins(%0) -> <P0xf16>
```

Comment:

Can we have a case like:

// P0=1024, R0=4096, P1=1024
input: <P0 x R0 x P1>
output: <P1 x P0>

I basically want to see if we are accounting for output coalescing as well. It
would be interesting to see what we should pick here. I don't know the answer
very well actually. We could take the hit on output coalescing by increasing
parallelism on the input by using more workgroups or we could coalesce the
output and get worse coalescing. I think the answer really depends on the
relative size of the input and output. If coalescing the output would increase
the input size significantly, we should probably not do it. But I don't know if
I know the answer here tbh. It would more just be a badly designed fused kernel.

**Phase A**:

Input `<R0xP0xf16>` (innermost=P0):
  contiguous_parallel_region = [P0]
  intrinsic_size = R0 = 4096 elements = 8KB → significant.
  Constraint: wg_tile[P0] >= 64 * 8 = 512.

Output `<P0xf16>` (innermost=P0):
  contiguous_parallel_region = [P0]
  intrinsic_size = 1 (no reduction dims in output) → 2 bytes. Tiny. Skip.

**Coalescing**: wg_tile[P0] >= 512.

**Phase B**:
- Tiling P0: affects both input and output. No broadcasts.

**Phase C**:
```
wg_tile = [P0=512]
numWGs = 1024 / 512 = 2
targetOccupancy = 1216

subgroupsPerWG = 16: 2 * 16 = 32. Still < 1216.
But maxWorkgroupSize = 1024, so subgroupsPerWG ≤ 16.
```

32 active subgroups across 2 WGs. Low occupancy, but each WG processes
4096 * 512 = 2M elements (4MB). Memory bandwidth is the bottleneck — even
2 WGs may saturate it.

Could reduce wg_tile[P0] to 128:
```
numWGs = 8. subgroupsPerWG = 16: 8*16 = 128.
Per-WG input: 4096 * 128 = 512K elements = 1MB.
128 / 8 = 16 threads along P0 for coalescing. Remaining 48 on R0.
```

**Per-WG shapes** (with P0=512):
```
input:  <4096 x 512 x f16> = 2M elements
output: <512 x f16>        = 512 elements
```

**Config**:
```
workgroup = [0, 512]
subgroupsPerWG = 16
// Threads coalesce along P0 (512 contiguous elements).
// R0 (4096 rows) is serial or partially distributed across subgroups.
```

Comment: When something like this happens where the coalescing constraint
restricts parallelism on a dimension, we should add a suggestion to the user
that doing this restricts parallelism and could be fixed if they fused the
kernel better.

---

### Example 3: Matvec

```
// P0=4(M), P1=4096(N), R0=4096(K)
// lhs: (d0,d1,d2) → (d0,d2) = <P0xR0xf16>
// rhs: (d0,d1,d2) → (d2,d1) = <R0xP1xf16>
// out: (d0,d1,d2) → (d0,d1) = <P0xP1xf16>
```

**Phase A**:

lhs `<P0xR0xf16>` (innermost=R0):
  contiguous_parallel_region = [] (R0 is reduction). No constraint.

rhs `<R0xP1xf16>` (innermost=P1):
  contiguous_parallel_region = [P1]
  intrinsic_size = R0 = 4096 → 8KB. Significant.
  Constraint: wg_tile[P1] >= 512.

out `<P0xP1xf16>` (innermost=P1):
  contiguous_parallel_region = [P1, P0]
  intrinsic_size = 1 (no reduction dims in output) → tiny. Skip.
  (After tiling, per-WG output is still small.)

**Coalescing**: wg_tile[P1] >= 512.

**Phase B**:

Tiling P0 (M=4):
  affected: lhs (has P0), out (has P0)
  unaffected: rhs (P0 not in rhs!) → rhs is broadcast across P0
  broadcast_cost(P0) = size(rhs) = 4096*4096 = 16M elements = 32MB

Tiling P1 (N=4096):
  affected: rhs (has P1), out (has P1)
  unaffected: lhs (P1 not in lhs!) → lhs is broadcast across P1
  broadcast_cost(P1) = size(lhs) = 4*4096 = 16K elements = 32KB

Key insight: **broadcast_cost(P0) >> broadcast_cost(P1)**.
Tiling P0 aggressively means many WGs each reading the full rhs (32MB
broadcast). Tiling P1 aggressively means many WGs each reading lhs (32KB
broadcast — fits in L2, cheap).

→ **Prefer tiling P1 over P0.** Keep P0 coarse.

**Phase C**:
```
wg_tile = [P0=1, P1=512]   // P1 from coalescing constraint
numWGs = 4 * (4096/512) = 4 * 8 = 32
targetOccupancy = 1216

subgroupsPerWG = 16: 32 * 16 = 512. Still < 1216.
```

Should we tile P0 further? P0=4, so wg_tile[P0] = 1 → 4 WGs along P0.
If we coarsen P0 (wg_tile[P0]=4): numWGs = 1 * 8 = 8. Fewer WGs, but
each WG handles all 4 M rows → can reuse rhs across M rows within a WG.

Actually, coarsening P0 REDUCES broadcast:
- wg_tile[P0]=1: 32 WGs, each reads full rhs → 32 * 32MB = 1GB total rhs traffic
- wg_tile[P0]=4: 8 WGs, each reads full rhs → 8 * 32MB = 256MB total rhs traffic
  (but lhs per WG is 4*4096 = 16K instead of 1*4096 = 4K — negligible)

So: **wg_tile[P0] = 4** (coarsen to reduce rhs broadcast).

```
wg_tile = [P0=4, P1=512]
numWGs = 1 * 8 = 8
subgroupsPerWG = 16: 8 * 16 = 128.
```

**Per-WG shapes**:
```
lhs: <4 x 4096 x f16>    = 16K elements (full lhs — it's tiny)
rhs: <4096 x 512 x f16>  = 2M elements  (dominant operand)
out: <4 x 512 x f16>     = 2K elements
```

**Config**:
```
workgroup = [4, 512, 0]
subgroupsPerWG = 16
// Threads coalesce along P1 (512 contiguous elements of rhs).
// R0 (K=4096) is reduction, distributed across subgroups.
// P0 (M=4) is batch per thread (serial over 4 rows).
```

---

### Example 4: Softmax

```
// P0=512, R0=10240
%0 = input <P0xR0xf32>
%2 = reduce(R0) ins(%0) -> <P0xf32>
%3 = elementwise ins(%0, %2) -> <P0xR0xf32>
%4 = reduce(R0) ins(%3) -> <P0xf32>
%5 = elementwise ins(%3, %4) -> <P0xR0xf32>
```

**Phase A**:

input/output `<P0xR0xf32>` (innermost=R0):
  contiguous_parallel_region = [] (R0 is reduction). No constraint.

max/sum `<P0xf32>`: tiny. Skip.

**No coalescing constraints.**

**Phase B**:
- Tiling P0: affects all operands. No broadcasts.

**Phase C**:
```
wg_tile = [P0=1]
numWGs = 512
subgroupsPerWG = 4 → workgroupSize = 256
512 * 4 = 2048 > 1216. ✓
```

**Per-WG shapes**:
```
input:  <1 x 10240 x f32> = 10240 elements = 40KB
output: <1 x 10240 x f32> = 10240 elements = 40KB
max:    <1 x f32>          = scalar
sum:    <1 x f32>          = scalar
```

---

### Example 5: Reduce-N in NHWC (multiple contiguous parallel dims)

```
// R_N=64, P_H=56, P_W=56, P_C=256
// input:  <R_N x P_H x P_W x P_C x f16>  (P_C innermost)
// output: <P_H x P_W x P_C x f16>
```

**Phase A**:

input `<R_N x P_H x P_W x P_C x f16>` (innermost=P_C):
  contiguous_parallel_region = [P_C, P_W, P_H]
  (R_N is outermost — doesn't break anything since we walk inner→outer)
  intrinsic_size = R_N = 64 → 128 bytes. Small? Borderline.
  Actually, the FULL tensor is 64*56*56*256 = 51M elements. After tiling
  parallel groups to 1, per-WG = 64*1*1*1 = 64 elements. Tiny!
  But with coalescing tiles: per-WG = 64 * product(coalescing_tiles).
  The point of coalescing is to make per-WG large enough to be useful.

  Constraint: wg_tile[P_C] * wg_tile[P_W] * wg_tile[P_H] >= 64 * 8 = 512.

output `<P_H x P_W x P_C x f16>` (innermost=P_C):
  contiguous_parallel_region = [P_C, P_W, P_H] (all parallel)
  intrinsic_size = 1 (no reduction dims). Tiny without WG tiles.
  But: output total = 56*56*256 = 802K elements = 1.6MB. Significant!

  For output coalescing, the question is whether each WG's output slice
  is written contiguously. With wg_tile[P_C]=c, wg_tile[P_W]=w:
  per-WG output = P_H_tile * P_W_tile * P_C_tile elements.
  If the output region is contiguous in memory: yes, coalesced writes.

  Same constraint: wg_tile product >= 512.

**Filling the constraint** (innermost first):
```
target = 512
wg_tile[P_C] = min(256, 512) = 256.  remaining = 512/256 = 2
wg_tile[P_W] = min(56, 2) = 2.       remaining = 2/2 = 1
wg_tile[P_H] = 1.                     done.

Product = 256 * 2 * 1 = 512 ✓
```

**Phase B**:
- All parallel groups affect all operands (no broadcasts).

**Phase C**:
```
wg_tile = [P_H=1, P_W=2, P_C=256]
numWGs = 56 * 28 * 1 = 1568
targetOccupancy = 1216
1568 > 1216 → enough WGs ✓

subgroupsPerWG = 1 (already enough WGs)
workgroupSize = 64
```

**Per-WG shapes**:
```
input:  <64 x 1 x 2 x 256 x f16> = 32K elements = 64KB
output: <1 x 2 x 256 x f16>      = 512 elements = 1KB
```

Thread distribution: 256/8 = 32 threads on P_C, 2 threads on P_W = 64.
Each thread loads 8 contiguous f16 along C, threads span C then W.
Reduction over R_N is serial per thread (64 iterations).

---

### Example 6: Batch Norm NHWC (parallel dim = C only)

```
// R_N=64, P_C=256, R_H=56, R_W=56
// input:  <R_N x R_H x R_W x P_C x f16>  (P_C innermost)
// output: <P_C x f16>
```

**Phase A**:

input (innermost=P_C):
  contiguous_parallel_region = [P_C] (R_W is next, breaks the chain)
  intrinsic_size = R_N * R_H * R_W = 64*56*56 = 200704 → 400KB. Significant!
  Constraint: wg_tile[P_C] >= 512. But P_C=256 < 512.
  → wg_tile[P_C] = 256 (capped at group size). Partial coalescing.

**Phase C**:
```
wg_tile = [P_C=256]
numWGs = 256 / 256 = 1. Only 1 WG!
subgroupsPerWG = 16 → 1*16 = 16. Still very low.
```

This is inherently low-parallelism. With only C=256 parallel and coalescing
needing all 256, there's just 1 WG. Subgroup scaling helps within the WG
but can't create more WGs.

**Per-WG shapes**:
```
input:  <64 x 56 x 56 x 256 x f16> = 51M elements (the entire tensor!)
output: <256 x f16>                 = 256 elements
```

Each WG processes the full input. 16 subgroups * 64 threads = 1024 threads
cooperate on this. Threads coalesce along P_C (256/8 = 32 threads on C).
Remaining 32 threads spread into reduction dims (R_W, R_H) for partial
cooperation.

---

## Algorithm Summary

```
function computeWorkgroupTiling(groups, tensors, target):
  // Phase A: Coalescing constraints
  wg_tile = {G: 1 for G in parallel_groups}
  for each tensor T:
    region = contiguousParallelRegion(T)
    if region.empty() or not isSignificant(T):
      continue
    target_size = subgroupSize * vectorWidth(T)
    remaining = target_size
    for G in region:  // innermost first
      needed = min(G.size, remaining)
      wg_tile[G] = max(wg_tile[G], needed)
      remaining = ceil(remaining / needed)
      if remaining <= 1: break

  // Phase B: Dimension interactions
  for each parallel group G:
    compute affected_operands(G), broadcast_cost(G)

  // Phase C: WG count and scaling
  numWGs = product(G.size / wg_tile[G] for G in parallel_groups)

  // C.3: Scale subgroups
  subgroupsPerWG = 1
  while numWGs * subgroupsPerWG < targetOccupancy:
    subgroupsPerWG *= 2
    if subgroupsPerWG * subgroupSize > maxWorkgroupSize:
      subgroupsPerWG /= 2; break

  // C.4: Coarsen if too many WGs
  while numWGs > coarsenFactor * targetOccupancy:
    G = argmax(broadcast_cost(G) for G where wg_tile[G]*2 <= G.size
               and G not critical for coalescing)
    wg_tile[G] *= 2
    numWGs /= 2

  return wg_tile, subgroupsPerWG
```
