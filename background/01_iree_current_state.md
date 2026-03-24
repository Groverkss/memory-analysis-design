# IREE Current Reduction Configuration

## Source
`compiler/src/iree/compiler/Codegen/Dialect/GPU/TargetUtils/ReductionConfigUtils.cpp`

## What it does

`setReductionConfig` is the entry point. It:
1. Validates the dispatch is compatible with VectorDistribute pipeline
2. Determines workgroup size, subgroup size, and thread loads
3. Attaches `lowering_config` attributes to linalg ops in the dispatch

## Config attributes produced

```mlir
#config = #iree_gpu.lowering_config<{
  workgroup = [...],
  partial_reduction = [...],   // or "serial" for parallel-only ops
  thread = [...],
  lane_basis = [[counts], [mapping]],
  subgroup_basis = [[counts], [mapping]]
}>
```

- **workgroup**: tile sizes for workgroup-level tiling (parallel dims shared across ops)
- **partial_reduction**: tile sizes for the reduction dimension tiling
- **thread**: per-thread vector sizes (i.e., how many elements each thread processes)
- **lane_basis**: `[counts, mapping]` - thread distribution within a subgroup
- **subgroup_basis**: `[counts, mapping]` - subgroup distribution

## How parameters are chosen

### Thread loads (vectorization width)
```
threadLoads = maxLoadBits / bitWidth     (e.g., 128 / 16 = 8 for f16)
```
Then adjusted down until:
- `(reductionSize / threadLoads) % subgroupSize == 0`
- All compute ops' constraints are satisfied (last dim divisible by threadLoads)

### Workgroup size
```
workgroupSize = reductionSize / threadLoads
```
Capped at `maxWorkgroupSize`. If enough parallel work exists to saturate GPU,
shrinks to a single subgroup.

Further reduced if:
- Parallel size > 256 (enough workgroups)
- AND each thread reads < 8 vectors (not enough work per thread)

### Subgroup size
First subgroup size choice from target where `reductionSize % subgroupSize == 0`.

### Partial reduction tile size
```
partialReductionSize = GCD(workgroupSize * threadLoads, reductionDimSize)
```
This is the chunk of the reduction dimension processed by the entire workgroup
in one "round".

### Thread/subgroup basis
```
threadBasis = subgroupSize  (start)
subgroupStride = threadBasis * threadLoads
// shrink threadBasis until partialReductionSize % subgroupStride == 0
subgroupBasis = partialReductionSize / subgroupStride
```

## Key assumptions / limitations

1. **Always focuses on the LAST reduction dimension** - the innermost reduction
   dim is assumed to be the one to distribute across threads.

2. **Single combiner op required** - multi-output reductions not supported.

3. **Expand dims trick**: When threadLoads > 1 and indexing maps are projected
   permutations, splits the last reduction dim into [outer, inner] where
   inner = threadLoads. This enables vectorized loads.

4. **No awareness of parallel dim layout** - parallel dims just get workgroup
   tile size = 1 (or shared). No thread distribution along parallel dims for
   the reduction op itself.

5. **Hardcoded thresholds**: parallelThreshold=256, targetVectorCount=8.

6. **Only handles reductions** - not sort, scan, topk, etc.

## Pipeline consumption

```
ReductionConfig → GPUApplyTilingLevel (workgroup, reduction, partial_reduction, serial)
  → LLVMGPUConfigureTensorLayouts (basis → NestedLayoutAttr)
  → Vectorization
  → LLVMGPUVectorDistribute (vector ops → SIMT distribution)
  → gpu.subgroup_reduce for cross-thread reductions
```
