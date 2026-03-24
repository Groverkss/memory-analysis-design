# Triton's Approach to Reductions & Memory-Bound Ops

## User-Level Patterns

### Softmax (row-wise reduction, single-pass)
```python
# Each program handles one row. All threads in a block cooperate on one row.
row_idx = tl.program_id(0)
col_offsets = tl.arange(0, BLOCK_SIZE)
row = tl.load(row_ptr + col_offsets, mask=col_offsets < n_cols, other=-inf)
row_max = tl.max(row, axis=0)          # warp shuffle reduce
numerator = tl.exp(row - row_max)
denominator = tl.sum(numerator, axis=0) # warp shuffle reduce
result = numerator / denominator
```

Key: entire row fits in registers. BLOCK_SIZE = next_power_of_2(n_cols).
num_warps scales with BLOCK_SIZE (typically BLOCK_SIZE // 256, clamped to [1, 8]).

### Layer Norm (row-wise reduction, multi-pass/chunked)
```python
# When N > BLOCK_SIZE, iterate in chunks
_mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
for off in range(0, N, BLOCK_SIZE):
    cols = off + tl.arange(0, BLOCK_SIZE)
    a = tl.load(X + cols, mask=cols < N, other=0.)
    _mean += a                              # accumulate in registers
mean = tl.sum(_mean, axis=0) / N           # final cross-thread reduce
```

Key: two passes needed (mean, then variance). Each pass iterates over chunks.
BLOCK_SIZE = min(65536 / element_size, next_power_of_2(N)).

### Backward (cross-row parallel reduction)
For dW/dB gradients, multiple rows need to contribute to the same weight gradient:
- GROUP_SIZE_M independent buffers with atomic locks
- Stage 1: each row accumulates into its group buffer (with locking)
- Stage 2: separate kernel reduces across groups

## Tile Size Selection Heuristics

| Operation | BLOCK_SIZE strategy |
|-----------|-------------------|
| Softmax | `next_power_of_2(n_cols)` - entire row in one block |
| Layer Norm | `min(64KB / elem_size, next_power_of_2(N))` |
| General | Power of 2, occupancy-driven |

**num_warps**: `min(max(BLOCK_SIZE // 256, 1), 8)`
**num_stages**: 4 if shared memory > 200KB, else 2

## Compiler-Level Reduction Lowering

Three-phase strategy:

### Phase 1: reduceWithinThreads
- Each thread reduces its own elements along the reduction axis
- If multiple elements per thread on reduction axis, uses tree reduction
- **Vectorization**: packs pairs into 2-lane vectors when hardware supports it

### Phase 2: reduceWithinWarps (warp shuffles)
- Uses `shuffleXor` to exchange values between threads within a warp
- Iterates through bit positions of `reduceLaneIdMask`
- Hardware REDUX instructions preferred when available (NVVM)

### Phase 3: reduceAcrossWarps (shared memory)
- Multi-round reduction using shared memory + barriers
- Intermediate layouts computed by `getInterLayout()`
- At most 2 rounds of inter-warp reductions

## Layout System (LinearLayout)

Triton uses a linear algebra-based layout system:
- Maps `(register, lane, warp, block)` → tensor indices
- Basis vectors at power-of-2 positions determine full layout via XOR linearity
- `BlockedLayout` has 4 levels: `size_per_thread`, `threads_per_warp`, `warps_per_cta`, `order`

### Coalescing
```
BlockedLayout(
  size_per_thread=[1, 4],       # 4 contiguous elements per thread in dim 1
  threads_per_warp=[1, 32],     # all 32 threads spread along dim 1
  warps_per_cta=[4, 1],         # warps spread along dim 0
  order=[1, 0],                 # dim 1 is innermost (contiguous in memory)
)
```

This ensures consecutive threads read consecutive memory addresses (coalesced).

For **row-wise reduction** (reduce dim 1):
- Threads are spread along dim 1 → coalesced reads
- Reduction via shuffles along dim 1

For **column-wise reduction** (reduce dim 0):
- Naive: threads spread along dim 0 → strided, uncoalesced
- Better: transpose → row-reduce → transpose back

## Scan Lowering

Three stages:
1. Within-thread sequential scan of contiguous elements
2. Within-warp parallel scan using `shuffleUp`
3. Multi-warp scan via shared memory (warp accumulators stored/loaded)

## Key Takeaways for IREE

1. **Layout drives everything**: Triton's layout system (BlockedLayout/LinearLayout)
   determines both memory access patterns and reduction strategy. The layout is
   chosen to maximize coalesced access first, then reduction falls out naturally.

2. **Thread distribution along reduction dim**: Consecutive threads handle
   consecutive elements of the reduction dimension → coalesced loads, then
   shuffle-reduce.

3. **Chunked iteration for large reductions**: When reduction dim > BLOCK_SIZE,
   iterate in chunks, accumulate in registers, final cross-thread reduce.

4. **Separate parallel dims from reduction dims**: Parallel dims map to
   program_id (workgroup), reduction dims map to thread distribution within
   the workgroup.

5. **Vectorization is per-thread**: Each thread loads multiple contiguous
   elements (size_per_thread), giving vectorized memory access.
