# Coalescing: Operand Level vs Iteration Space Level

## The key distinction

Coalescing is about **memory access patterns of operands**, not about the
iteration space. The iteration space is an abstraction; what matters is how
threads map to actual memory addresses when they load/store operand tensors.

## DerivedThreadConfigAttr gets this wrong

`DerivedThreadConfigAttr` distributes threads from innermost iteration dim
outward. This only gives coalesced access when the indexing maps are identity
(or at least preserve the innermost dim ordering). With a transposed operand,
it produces strided access.

Example:
```mlir
linalg.generic {
  indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>,   // input: transposed
                   affine_map<(d0, d1) -> (d0, d1)>],  // output: identity
  iterator_types = ["parallel", "parallel"]
}
```

If threads are distributed along d1 (innermost iteration dim):
- Output access: `output[*, thread_id]` → contiguous ✓
- Input access: `input[thread_id, *]` → strided ✗

If threads are distributed along d0:
- Output access: `output[thread_id, *]` → strided ✗
- Input access: `input[*, thread_id]` → contiguous ✓

Can't satisfy both. Must choose which operand to prioritize.

## The right approach

1. **Identify which operand to coalesce for**
   - Usually the largest operand (most memory traffic)
   - Or the operand that is read most frequently (e.g., in a reduction loop,
     the input is read many times but output is written once)
   - For reductions: the input tensor dominates memory traffic

2. **Find the contiguous dimension of that operand**
   - For standard row-major layout: last dimension is contiguous
   - Could be different with non-standard layouts

3. **Map back through the indexing map to iteration space**
   - If `indexing_map = (d0, d1) -> (d1, d0)`, and operand dim 1 is contiguous,
     then iteration dim d0 maps to operand dim 1 → distribute threads along d0

4. **Set thread distribution (lane_basis) accordingly**

## Multiple operands with conflicting needs

When operands have different indexing maps, there's a conflict. Options:

a. **Prioritize the dominant operand** (largest / most accessed)
   - Simple, works well for most cases
   - For reductions: prioritize input (read in a loop) over output (written once)

b. **Use shared memory** for the non-prioritized operand
   - Load with coalesced access into shared memory
   - Read from shared memory with the layout the computation needs
   - This is what Triton/IREE already do for matmul (promote operands)

c. **Choose based on total bandwidth impact**
   - Estimate bytes loaded/stored for each choice
   - Pick the distribution that minimizes total uncoalesced traffic

## What this means for the config

The lowering_config should carry information about which operand (or which
dimension of which operand) to optimize coalescing for. This could be:

### Option A: Explicit coalesced operand index
```mlir
#config = #iree_gpu.lowering_config<{
  ...,
  coalesced_operand = 0    // optimize thread layout for operand 0's access
}>
```
Then ConfigureTensorLayouts can look at operand 0's indexing map + tensor type
to determine the optimal thread distribution.

### Option B: Explicit coalesced iteration dim
```mlir
#config = #iree_gpu.lowering_config<{
  ...,
  coalesced_dim = 1    // distribute threads to coalesce iteration dim 1
}>
```
This is less general (doesn't account for operand layouts) but simpler.
The config generator would compute this from the operand analysis.

### Option C: Just get the basis right in the config generator
The lane_basis already determines thread distribution. If the config generator
(ReductionConfigUtils.cpp) analyzes the operand indexing maps and memory layouts
correctly, it can set the lane_basis mapping to achieve coalesced access without
any new config field.

The new field would be purely for readability/debuggability — making the intent
explicit rather than having it be an implicit property of the basis.

## For reductions specifically

Consider `tensor<M x N xf16>`:

### Case 1: Reduce dim N (innermost, contiguous)
- Coalesce for: input, dim N is contiguous
- Map N → iteration dim (say d1): distribute threads along d1
- This is what the current config already does ✓

### Case 2: Reduce dim M (outermost, strided)
- Coalesce for: input, dim N is contiguous (even though N is parallel!)
- Map N → iteration dim (say d1): distribute threads along d1
- Reduction over M (d0) becomes serial loop per thread
- lane_basis mapping: threads → d1 (parallel dim, but that's fine!)

### Case 3: Reduce dim M of tensor<M x N> with transposed input
```
indexing_map = (d0, d1) -> (d1, d0)
```
- Operand memory: dim 1 of operand = d0 is contiguous
- Distribute threads along d0 → coalesced input reads
- Reduction over M (d0) is distributed across threads → need shuffle reduce
- Parallel dim N (d1) is serial/workgroup-level

This is exactly like Case 1 but through a transposed lens. The indexing map
inverts the relationship.

## Bottom line

The config generator needs to:
1. Look at operand types and indexing maps (not just iterator types)
2. Determine which iteration dim gives coalesced access for the dominant operand
3. Put threads along THAT dim via lane_basis, regardless of parallel/reduction
4. The downstream pipeline handles the rest correctly already
