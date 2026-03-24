# Comparison: IREE vs Triton & Gaps

## Terminology Mapping

| IREE | Triton |
|------|--------|
| workgroup tile | program/CTA tile |
| subgroup | warp |
| thread (lane) | thread (lane) |
| threadLoads | size_per_thread (on reduction axis) |
| partial_reduction tile | BLOCK_SIZE (reduction chunk) |
| lane_basis / subgroup_basis | BlockedLayout (threads_per_warp, warps_per_cta) |

## What IREE does similarly

- Thread vectorization: `threadLoads` = elements per thread = Triton's `size_per_thread`
- Subgroup-level distribution: `threadBasis` threads within subgroup on reduction dim
- Multi-subgroup: `subgroupBasis` subgroups cooperating on reduction
- Chunked reduction: `partial_reduction` tile enables iterating over reduction dim

## What IREE is missing / doing differently

### 1. Only handles "innermost reduction dim" well

IREE's config always distributes threads along the **last reduction dim**.
This works for:
- `sum(x, dim=-1)` where dim -1 is contiguous in memory → coalesced

But fails for:
- `sum(x, dim=0)` on a row-major tensor → threads read strided memory
- Batch norm: reduce over batch dim (outermost) → strided

**Triton's answer**: Layout chooses thread order to match memory layout.
If reducing dim 0 of a row-major tensor, Triton would either:
- Assign threads to columns, iterate over rows (each thread reads strided, but
  multiple threads read same cache line) - or -
- Transpose first for better access

### 2. No awareness of memory layout in config

IREE's reduction config doesn't consider which dimension is contiguous in memory.
It only looks at the linalg iterator types (parallel vs reduction) and dimension
sizes.

**What should happen**: The config should know that for a `tensor<M x N x f16>`
with default layout, dim N is contiguous. If reducing over dim N, threads should
be distributed along N. If reducing over dim M, the strategy should be different
(potentially transposing, or accepting strided access with appropriate tile sizes).

### 3. Parallel dim distribution is an afterthought

For parallel dims, IREE just sets workgroup tile = 1 and lets them be serial.
This means:
- For `softmax(tensor<1024x4096>)`: workgroup handles 1 row, threads reduce
  along 4096. Good.
- For `batchnorm(tensor<64x256x56x56>)`: reducing over batch(0),H(2),W(3),
  keeping C(1). Current config doesn't handle this well because the "reduction"
  is over non-contiguous dims.

**Triton's answer**: Each program_id maps to one output element (or tile of
output elements). Within the program, threads cooperate on the reduction.
The parallel-to-workgroup mapping is explicit.

### 4. No multi-dimensional thread distribution

IREE's basis is 1D: threads are distributed along a single dimension.
For a 2D reduction like `tensor<M x K> -> tensor<M>`:
- All threads go along K (reduction)
- M is purely workgroup-level

But for operations where we want threads distributed across BOTH dimensions
(e.g., 2D tiles for better cache utilization), the current config can't express
that for the reduction op itself.

### 5. Missing operation classes

Current config only handles linalg reductions with single combiner.
Not supported:
- **argmax/argmin**: multi-output reduction (value + index)
- **sort**: not a simple reduction
- **topk**: combination of sort + take
- **scan/cumsum**: prefix sums, different access pattern
- **multi-combiner**: e.g., simultaneous mean + variance

## Summary of what needs to change

The core issue is that "reduction configuration" is really about:
**"How do we tile and distribute a memory-bound operation across threads
to maximize memory throughput?"**

The answer depends on:
1. Which dimensions are reduced vs parallel
2. Which dimension is contiguous in memory
3. The sizes of each dimension
4. Hardware constraints (subgroup size, max workgroup size, vector width)

The new config should:
- Consider memory layout when distributing threads
- Support multi-dimensional thread distribution
- Handle both innermost and outermost reductions
- Be extensible to sort/scan/topk operations
- Name: something like "MemoryBoundConfig" rather than "ReductionConfig"
