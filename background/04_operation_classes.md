# Operation Classes & Their Access Patterns

## Class 1: Innermost-dim Reduction
Examples: layer norm, softmax, row-wise sum/max

```
input:  tensor<M x N>  (N contiguous)
output: tensor<M>
reduce: dim N (contiguous)
```

**Ideal strategy**:
- Workgroup per row (or tile of rows)
- Threads distributed along N (contiguous → coalesced loads)
- Each thread loads `vectorWidth` contiguous elements
- Shuffle-reduce across threads within subgroup
- If N > workgroupSize * vectorWidth: chunked loop, accumulate in registers

**Triton equivalent**:
```python
row = tl.program_id(0)
cols = tl.arange(0, BLOCK_N)
x = tl.load(ptr + row * N + cols)
result = tl.sum(x, axis=0)
```

**IREE status**: Works well today. This is what the current config targets.


## Class 2: Outermost-dim Reduction
Examples: batch norm (reduce over batch), column-wise sum

```
input:  tensor<M x N>  (N contiguous)
output: tensor<N>
reduce: dim M (strided)
```

**Ideal strategy**:
- Workgroup per tile of output (tile of columns)
- Threads distributed along N (output dim, contiguous) for coalesced reads
- Each thread handles a column (or a few columns)
- Loop over M rows sequentially, accumulating in registers
- No cross-thread reduction needed if each thread owns its own column(s)!

**Triton equivalent**:
```python
col_block = tl.program_id(0)
cols = col_block * BLOCK_N + tl.arange(0, BLOCK_N)
acc = tl.zeros([BLOCK_N], dtype=tl.float32)
for row in range(M):
    x = tl.load(ptr + row * N + cols, mask=cols < N)
    acc += x
tl.store(out + cols, acc)
```

**Key insight**: This is NOT a "distribute threads along reduction dim" problem.
Threads should be along the PARALLEL (output) dim for coalesced access.
The reduction (over rows) is a serial loop per thread.

**IREE status**: The current config would try to distribute threads along M
(the reduction dim), which is wrong for memory access. This is the biggest gap.


## Class 3: Multi-dim Reduction (batch norm style)
Examples: batch norm (reduce over N,H,W keeping C)

```
input:  tensor<N x C x H x W>  (W contiguous)
output: tensor<C>
reduce: dims N, H, W
parallel: dim C
```

**Ideal strategy**:
- Workgroup per C (or tile of C)
- Threads distributed to maximize coalesced access
- Since W is contiguous, want threads along W
- Loop over N and H, accumulate in registers
- Cross-thread reduce across W tiles

This is a hybrid: some reduction dims are sequential (N, H), innermost (W)
is distributed across threads.

**IREE status**: Not well handled. The config sees 3 reduction dims and would
try to distribute along the last one, but doesn't understand the memory layout
implications.


## Class 4: Argmax/Argmin
```
input:  tensor<M x N>
output: tensor<M> (indices), tensor<M> (values)
reduce: dim N
```

Same access pattern as Class 1, but multi-output reduction.
Each thread tracks both value and index, shuffle-reduce needs custom combiner.

**IREE status**: Fails the `checkSingleCombiner` constraint.


## Class 5: Scan/Cumsum
```
input:  tensor<M x N>
output: tensor<M x N>  (same shape)
scan:   dim N (or dim M)
```

**Ideal strategy (scan along contiguous dim N)**:
- Phase 1: each thread scans its contiguous elements sequentially
- Phase 2: cross-thread parallel scan via shuffles (Blelloch/Hillis-Steele)
- Phase 3: broadcast prefix back to update thread-local values

**Ideal strategy (scan along strided dim M)**:
- Threads distributed along N for coalesced access
- Each thread loops over M sequentially, maintaining running state

**IREE status**: Not supported at all in reduction config.


## Class 6: Sort
```
input:  tensor<M x N>
output: tensor<M x N>  (sorted along some dim)
```

Typically bitonic sort within a block, merge sort across blocks.
Access pattern similar to reduction but with data movement.

**IREE status**: Not supported.


## Class 7: TopK
```
input:  tensor<M x N>
output: tensor<M x K>  (K << N)
```

Partial sort / selection. Can be done with a heap per thread or
parallel partial sort.

**IREE status**: Not supported.


## Class 8: Matvec (memory-bound contraction)

```
lhs:    tensor<M x K>      (M is small, e.g., 1-8)
rhs:    tensor<K x N>      (the big matrix)
output: tensor<M x N>
contract: K is reduction, M and N are parallel
```

Matvec is a contraction but memory-bound (not compute-bound like matmul)
because the arithmetic intensity is O(1) — each matrix element is used only
once. The matrix (rhs) dominates memory traffic.

**Key property**: lhs is broadcast across N, rhs is broadcast across M.
- Parallelizing N shrinks rhs per WG but NOT lhs
- Parallelizing M shrinks lhs per WG but NOT rhs
- rhs is dominant (K×N elements vs M×K for lhs, and M is small)
- The tradeoff: more N-WGs → more redundant lhs loads, less rhs per WG

**Ideal strategy**:
- Tile N across workgroups (each WG handles N_tile output columns)
- Within a WG, threads distributed across BOTH N (coalescing for rhs) and
  K (reduction cooperation)
- lhs is small enough to fit in registers/L2 — load once, reuse across N
- Shuffle-reduce across threads on K

**Variants**:
- Batched matvec: tensor<B x 1 x K> × tensor<B x K x N> — batch dim is
  trivially parallel
- Multi-head attention's QK^T when seq_len is small
- Any matmul where one dimension is very small (M=1 or small M)

**IREE status**: Has a special case in ReductionConfigUtils.cpp (ROCm only)
that splits the parallel dim. The new design handles this generically through
parallelism + coalescing analysis.


## Unifying Principle

All these operations are **memory-bound**. The key optimization is:
1. **Coalesced global memory reads** - threads read consecutive addresses
2. **Minimize global memory traffic** - process as much as possible in
   registers/shared memory before writing back
3. **Appropriate thread distribution** - based on memory layout, not
   iterator type

The thread distribution should be driven by: "which dimension is contiguous
in memory?" rather than "which dimension is a reduction?"
