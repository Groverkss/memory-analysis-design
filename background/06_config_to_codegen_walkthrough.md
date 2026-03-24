# End-to-End Walkthrough: Config → Codegen

## How the pieces fit together

```
ReductionConfigUtils.cpp          → produces lowering_config dict
  ↓
GPUApplyTilingLevel               → consumes workgroup/partial_reduction/serial tile sizes
  ↓
LLVMGPUConfigureTensorLayouts     → consumes lane_basis/subgroup_basis → NestedLayoutAttr
  ↓
LLVMGPUVectorDistribute           → consumes NestedLayoutAttr → distributed vector ops
```

## Detailed Example: f16 reduction of tensor<1024x4096>

### Input
```mlir
linalg.generic {
  indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,  // input
                   affine_map<(d0, d1) -> (d0)>],      // output
  iterator_types = ["parallel", "reduction"]
} ins(%in : tensor<1024x4096xf16>)
  outs(%out : tensor<1024xf16>) {
  ^bb0(%arg0: f16, %arg1: f16):
    %0 = arith.addf %arg0, %arg1 : f16
    linalg.yield %0 : f16
}
```

### Config Generation (ReductionConfigUtils.cpp)

```
reductionSize = 4096 (dim 1)
bitWidth = 16 (f16)
threadLoads = 128 / 16 = 8  (128-bit vector load)
subgroupSize = 64
workgroupSize = 4096 / 8 = 512, capped to 256 (maxWorkgroupSize)
  → further reduced: parallelSize=1024 > 256, and 256/2=128 still % 64 == 0
  → workgroupSize = 128 (2 subgroups)
  → actually let's say it stays at 256 for simplicity

partialReductionSize = GCD(256 * 8, 4096) = GCD(2048, 4096) = 2048
threadBasis = 64 (start), subgroupStride = 64 * 8 = 512
2048 % 512 == 0 ✓
subgroupBasis = 2048 / 512 = 4
```

Config on the reduction op:
```mlir
#config = #iree_gpu.lowering_config<{
  workgroup = [1, 0],
  partial_reduction = [0, 2048],
  thread = [0, 8],
  lane_basis = [[1, 64], [0, 1]],        // 64 threads along dim 1
  subgroup_basis = [[1, 4], [0, 1]]      // 4 subgroups along dim 1
}>
```

TranslationInfo: workgroup_size = [256, 1, 1], subgroup_size = 64

### After Workgroup Tiling

Each workgroup handles 1 row:
```mlir
// workgroup processes tensor<1x4096xf16> → tensor<1xf16>
```

### After Partial Reduction Tiling

Loop over 4096/2048 = 2 chunks:
```mlir
scf.for %k = 0 to 4096 step 2048 {
  // tile: tensor<1x2048xf16>
  linalg.generic ... // reduces 2048 elements
}
```

### After Vectorization

```mlir
scf.for %k = 0 to 4096 step 2048 {
  %v = vector.transfer_read ... : vector<1x2048xf16>
  %r = vector.multi_dim_reduction<add> %v, %acc [1] : vector<1x2048xf16> to vector<1xf16>
  ...
}
```

### LLVMGPUConfigureTensorLayouts

Reads config, constructs layout for vector<1x2048xf16>:

```
bounds = [1, 2048]

Subgroup distribution (subgroup_basis = [[1, 4], [0, 1]]):
  subgroupSizes = applyPermutation([1, 4], [0, 1]) = [1, 4]
  subgroupStrides = applyPermutation([4, 1], [0, 1]) = [4, 1]
  bounds = [1, 2048] / [1, 4] = [1, 512]

Thread distribution (lane_basis = [[1, 64], [0, 1]]):
  threadSizes = [1, 64]
  threadStrides = [64, 1]
  bounds = [1, 512] / [1, 64] = [1, 8]

Element tile (thread sizes = [0, 8] → [1, 8] effective):
  bounds = [1, 8] / [1, 8] = [1, 1]

Batch tile = bounds = [1, 1]
```

NestedLayout:
```
subgroup_tile = [1, 4]    subgroup_strides = [4, 1]
batch_tile    = [1, 1]
outer_tile    = [1, 1]
thread_tile   = [1, 64]   thread_strides = [64, 1]
element_tile  = [1, 8]
```

Check: 1*1*1*1*1 = 1 ✓ (dim 0)
       4*1*1*64*8 = 2048 ✓ (dim 1)

### Vector Distribution

The vector<1x2048xf16> gets distributed:
- 4 subgroups × 64 threads = 256 threads
- Each thread holds vector<1x8xf16> (8 contiguous f16 = 128-bit load)
- Threads 0-63 in subgroup 0 handle elements 0-511
- Threads 0-63 in subgroup 1 handle elements 512-1023
- etc.

**Transfer read**: Each thread does one 128-bit coalesced load.

**Multi-dim reduction**:
1. **Local reduce**: Each thread reduces its 8 elements → 1 scalar
2. **Thread reduce**: `gpu.subgroup_reduce(val, add, cluster_size=64, cluster_stride=1)`
   → butterfly shuffle across 64 threads in subgroup → each thread has subgroup partial sum
3. **Subgroup reduce**: Write to shared memory, barrier, read back, reduce
   → combines 4 subgroup results

## What Changes for Outermost-Dim Reduction

`tensor<4096x1024xf16> → tensor<1024xf16>` reducing dim 0 (strided)

Current config would try: threads along dim 0 → strided loads → bad.

Better config:
```mlir
#config = #iree_gpu.lowering_config<{
  workgroup = [0, 128],              // tile 128 columns per workgroup
  partial_reduction = [256, 0],      // 256 rows per chunk
  thread = [1, 4],                   // 4 contiguous elements (dim 1) per thread
  lane_basis = [[1, 64], [0, 1]],    // 64 threads along dim 1 (contiguous!)
  subgroup_basis = [[1, 1], [0, 1]]  // 1 subgroup
}>
```

After tiling: each workgroup handles tensor<4096x128xf16> → tensor<128xf16>
Loop: 4096/256 = 16 chunks of tensor<256x128xf16>

Layout for vector<256x128xf16>:
```
subgroup_tile = [1, 1]
batch_tile    = [256, 1]    // 256 rows = serial loop per thread
thread_tile   = [1, 64]     // 64 threads along cols
element_tile  = [1, 4]      // 4 contiguous f16 per thread (128-bit)

Wait, check: 1*256*1*1*1 = 256 ✓ (dim 0)
             1*1*1*64*4  = 256 ≠ 128... need to adjust.
```

Hmm, let me recalculate. With 64 threads × 4 elements = 256 columns, but we
only have 128. So either:
- Use 32 threads: lane_basis = [[1, 32], [0, 1]], thread = [1, 4]
  → 32 × 4 = 128 ✓
- Or use 64 threads with 2 elements: lane_basis = [[1, 64], [0, 1]], thread = [1, 2]
  → 64 × 2 = 128 ✓

Either way, the key insight is that threads are along dim 1 (contiguous in
memory), NOT along dim 0 (the reduction dim). The reduction over dim 0 happens
in the batch dimension — it's a serial accumulation loop, no cross-thread
communication needed for the reduction itself.

## DerivedThreadConfigAttr Comparison

`DerivedThreadConfigAttr` (used for non-reduction ops) distributes threads
from innermost dim outward:

```cpp
for (auto [tile, stride, size] :
     llvm::reverse(llvm::zip(threadTile, threadStrides, opShape))) {
  // Assign as many threads as possible to innermost dim first
}
```

This naturally produces coalesced access because the innermost dim is
contiguous in memory. The new reduction config should follow a similar
principle: **distribute threads starting from the contiguous (innermost)
dimension**, regardless of whether it's a parallel or reduction dim.

## Summary: What the Config Controls

| Config field | Pipeline consumer | Effect |
|---|---|---|
| workgroup | tileAndDistributeToWorkgroup | How work splits across workgroups |
| partial_reduction | GPUApplyTilingLevel | Chunk size for reduction loop |
| serial | GPUApplyTilingLevel | Additional serial tiling |
| thread | LLVMGPUConfigureTensorLayouts | Vector width per thread (element_tile) |
| lane_basis | LLVMGPUConfigureTensorLayouts | Thread distribution pattern (thread_tile + strides) |
| subgroup_basis | LLVMGPUConfigureTensorLayouts | Subgroup distribution pattern |

The downstream pipeline (LLVMGPUVectorDistribute) doesn't look at the config
at all — it only sees NestedLayoutAttr. So any changes to config logic are
fully contained in ReductionConfigUtils.cpp and how it populates these fields.

## Extensibility

Adding new config fields (e.g., `"coalesced"`) would:
1. Be set in ReductionConfigUtils.cpp
2. Be consumed in LLVMGPUConfigureTensorLayouts.cpp (or a new pass)
3. NOT require changes to the distribution patterns

But actually, the existing basis mechanism is already sufficient to express
any thread distribution pattern. The real change needed is in the CONFIG
GENERATION LOGIC — making ReductionConfigUtils.cpp smart enough to choose
the right basis based on memory layout, not just iterator types.
