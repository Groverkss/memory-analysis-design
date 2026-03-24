# VectorDistribute Pipeline: How lowering_config Drives Codegen

## Pipeline Stages

`addGPUVectorDistributePassPipeline` in `Codegen/LLVMGPU/Passes.cpp`:

```
1. tileAndDistributeToWorkgroup     -- workgroup-level tiling
2. GPUApplyTilingLevel(Reduction)   -- tile reduction dims
3. GPUApplyTilingLevel(PartialReduction) -- tile partial reduction
4. GPUApplyTilingLevel(Serial)      -- tile serial dims
5. Vectorization                    -- linalg → vector ops
6. LLVMGPUConfigureTensorLayouts    -- lowering_config → NestedLayoutAttr → ToLayoutOp
7. LLVMGPUVectorDistribute          -- vector ops → distributed SIMT code
8. Post-distribution cleanup/lowering
```

## LoweringConfigAttr Structure

Defined in `Codegen/Dialect/GPU/IR/IREEGPUAttrs.td`. It's a wrapper around
DictionaryAttr, so any named field can be added.

### Current fields used by reduction config

```mlir
#iree_gpu.lowering_config<{
  workgroup = [wg_0, wg_1, ...],           // workgroup tile sizes
  partial_reduction = [pr_0, pr_1, ...],    // partial reduction tile sizes
  thread = [t_0, t_1, ...],                // per-thread vector sizes
  lane_basis = [[counts], [mapping]],       // thread distribution (TilingLevel::Thread)
  subgroup_basis = [[counts], [mapping]],   // subgroup distribution (TilingLevel::Subgroup)
  expand_dims = <...>                       // optional dimension expansion
}>
```

### Tiling levels (IREEGPUEnums.td)

```
Workgroup → Reduction → PartialReduction → Serial → Thread → Subgroup → Lane
```

Each level maps to a key in the dict: `"workgroup"`, `"reduction"`,
`"partial_reduction"`, `"serial"`, `"thread"`, `"subgroup"`, `"lane"`.

### Basis struct (GPULoweringConfigUtils.h)

```cpp
struct Basis {
  SmallVector<int64_t> counts;   // resource count per dimension
  SmallVector<int64_t> mapping;  // projected permutation to iteration space
};
```

The basis describes distribution of a resource (subgroups or threads) across
the iteration space. Given a linear resource ID `x`:
```
b = delinearize(x, counts)
idx = apply(b, mapping)
```

The "lane_basis" key stores the basis for TilingLevel::Thread.
The "subgroup_basis" key stores the basis for TilingLevel::Subgroup.

### Adding new fields

Since it's a DictionaryAttr, adding a new field is just:
1. Pick a string key (e.g., `"coalesced"`)
2. Add helper functions in GPULoweringConfigUtils.h/.cpp
3. Include the NamedAttribute when building the config dict
4. Read it wherever needed in the pipeline

## Stage 1-4: Tiling (GPUApplyTilingLevel)

`Codegen/Common/GPU/GPUApplyTilingLevel.cpp`

For each tiling level, the pass:
1. Walks all ops with a lowering_config
2. Checks `config.hasTilingLevel(level)`
3. Gets tile sizes via `config.getStaticTilingLevelSizes(level, op)`
4. Applies tile-and-fuse

For reductions, this creates:
- Workgroup tiles along parallel dims
- A loop over the reduction dim with `partial_reduction` chunk size
- Within each chunk, `serial` tiling (if present)

After tiling, the linalg ops operate on tiles small enough for vectorization.

## Stage 6: LLVMGPUConfigureTensorLayouts

`Codegen/LLVMGPU/LLVMGPUConfigureTensorLayouts.cpp`

For ops with LoweringConfigAttr (non-MMA), calls `setGPULoweringConfigLayout`:

### Step 1: Get iteration space bounds
```cpp
SmallVector<int64_t> bounds = getIterationSpaceBounds(candidate);
// These are the post-tiling shape, e.g., [1, 256] for a reduction tile
```

### Step 2: Distribute subgroup basis
```cpp
distributeTilingSizes(candidate, config, TilingLevel::Subgroup,
                      bounds, subgroupSizes, subgroupStrides);
// Extracts subgroup_basis, applies mapping permutation
// bounds gets divided by subgroupSizes
```

### Step 3: Distribute thread basis
```cpp
distributeTilingSizes(candidate, config, TilingLevel::Thread,
                      bounds, threadSizes, threadStrides);
// Extracts lane_basis, applies mapping permutation
// bounds gets further divided
```

### Step 4: Compute element tile
```cpp
threadTileSizes = config.getStaticTilingLevelSizes(TilingLevel::Thread, op);
elementTile = divideTile(bounds, threadTileSizes);
// elementTile = what's left for each thread's native vector
```

### Step 5: Remaining = batch tile
```cpp
batchTile = bounds;  // whatever's left after subgroup/thread/element division
```

### Step 6: Create NestedLayoutAttr
```cpp
layout = NestedLayoutAttr::get(context,
    subgroupSizes, batchTile, outerTile=[1,...],
    threadSizes, elementTile,
    subgroupStrides, threadStrides);
```

### Step 7: Wrap operands and results with ToLayoutOp
```cpp
for (operand : candidate->getOpOperands()) {
    operandLayout = layout.apply(candidate.getMatchingIndexingMap(&operand));
    toLayout = ToLayoutOp::create(rewriter, loc, operand.get(), operandLayout);
    operand.set(toLayout);
}
// Similar for results
```

The `layout.apply(indexingMap)` projects the iteration-space layout through the
indexing map to get the operand/result layout. This is how a reduction dim that
doesn't appear in the output gets dropped from the output layout.

## NestedLayoutAttr

Defined in `Codegen/Dialect/VectorExt/IR/VectorExtAttrs.td`.

A vector of shape `[D0, D1, ...]` is viewed as a 5-level hierarchy:
```
[subgroup_tile] x [batch_tile] x [outer_tile] x [thread_tile] x [element_tile]
```

Where `D_i = subgroup[i] * batch[i] * outer[i] * thread[i] * element[i]`.

- **subgroup_tile + subgroup_strides**: Which subgroup owns which chunk
- **batch_tile**: Unrolled iterations (serial within a thread, across subgroups)
- **outer_tile**: Extra unrolling (usually 1 for non-MMA)
- **thread_tile + thread_strides**: Which thread within a subgroup
- **element_tile**: Native vector width per thread

Thread/subgroup IDs are mapped via strides:
```
virtual_thread_id[i] = (thread_id / thread_stride[i]) % thread_tile[i]
```

stride=0 means the dimension is NOT distributed (broadcast to all threads).

## Stage 7: DistributeMultiReduction

`Codegen/Common/GPU/GPUNestedLayoutDistributionPatterns.cpp`

The reduction distribution has 3 stages:

### Stage A: Local reduce
Each thread reduces its batch, outer, element dimensions locally:
```cpp
// Create reduction mask over all 3 copies of the reduction dims
// (batch copy, outer copy, element copy)
localReduction = vector.multi_dim_reduction(disSrc, localInit, mask, kind)
```

### Stage B: Thread reduce (subgroup shuffles)
If `threadTile[reductionDim] > 1`, threads within a subgroup need to
exchange values:
```cpp
for each element in the flattened local result:
    for each reduction dim where threadTile[dim] > 1:
        offset = layout.getThreadStrides()[dim]  // = shuffle stride
        width  = layout.getThreadTile()[dim]     // = cluster size
        extracted = gpu.subgroup_reduce(extracted, kind,
                                        cluster_size=width,
                                        cluster_stride=offset)
```

This becomes butterfly shuffles that reduce across `width` threads
spaced `offset` apart.

### Stage C: Subgroup reduce (shared memory)
If `subgroupTile[reductionDim] > 1`, subgroups cooperate via shared memory:
1. Write partial results to workgroup shared memory
2. Barrier
3. Read back with redistributed layout
4. Final local reduction

## Concrete Example: Row-wise Reduction

`linalg.generic {reduction_dim=1} : tensor<1x4096xf16> -> tensor<1xf16>`

Config might be:
```mlir
workgroup = [1, 0],
partial_reduction = [0, 256],   // 256 elements per workgroup chunk
thread = [0, 4],                // 4 elements per thread (f16 → 128-bit load)
lane_basis = [[64], [1]],       // 64 threads along dim 1
subgroup_basis = [[1], [1]]     // 1 subgroup
```

After tiling:
- Workgroup tile: 1 row
- Loop over 4096/256 = 16 chunks
- Each chunk: vector<1x256xf16>

Layout construction:
- subgroupSizes = [1, 1], subgroupStrides = [0, 1]
- threadSizes = [1, 64], threadStrides = [0, 1]
- elementTile = [1, 4]   (thread tile sizes)
- batchTile = [1, 1]     (256 / 1 / 64 / 4 = 1)

NestedLayout: `vector<1x256xf16>` decomposed as:
```
subgroup[1,1] x batch[1,1] x outer[1,1] x thread[1,64] x element[1,4]
```

Distribution:
- Each of 64 threads loads 4 contiguous f16 values (128-bit coalesced load)
- Local reduce: identity (nothing to reduce locally in batch/element since batch=1)
  Actually the reduction is along dim 1, so the element dim IS reduced locally.
  Each thread reduces its 4 elements to 1.
- Thread reduce: `gpu.subgroup_reduce(val, add, cluster_size=64, cluster_stride=1)`
  → butterfly shuffle across 64 threads

## Key Insight for New Config

The `lane_basis` determines thread distribution. Currently for reductions:
```
lane_basis = [[threadBasis], [lastReductionDim]]
```
Threads are always distributed along the last reduction dim.

For outermost-dim reduction (e.g., reducing dim 0 of tensor<M x N>),
we'd want:
```
lane_basis = [[threadBasis], [1]]   // threads along dim 1 (parallel, contiguous)
```
And the reduction over dim 0 would be a serial loop per thread.

The config already has the mechanism to express this via the `mapping` field
in the basis. The problem is entirely in ReductionConfigUtils.cpp's logic
for choosing WHAT to put in the basis, not in the downstream pipeline's
ability to consume it.

We could also add a `"coalesced"` field to explicitly communicate to
LLVMGPUConfigureTensorLayouts which dimension should get the densest
thread distribution for memory coalescing.
