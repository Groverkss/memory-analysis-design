# Liger-Kernel: Memory-Bound Triton Kernel Analysis

## Source: https://github.com/linkedin/Liger-Kernel.git

---

## Critical Foundation: `calculate_settings()` (used by most kernels)

**File:** `src/liger_kernel/ops/utils.py`, lines 45-62

```python
BLOCK_SIZE = triton.next_power_of_2(n)  # n = reduction dimension
# Cap at MAX_FUSED_SIZE = 65536
num_warps = 4                      # default
num_warps = 8   if BLOCK_SIZE >= 2048
num_warps = 16  if BLOCK_SIZE >= 8192
num_warps = 32  if BLOCK_SIZE >= 32768  (16 on HIP/AMD)
```

Key insight: **BLOCK_SIZE always equals `next_power_of_2(n_cols)`** -- it always covers the entire reduction dimension in a single tile. There is NO tiling across the reduction dimension in the "single-block" path. The `MAX_FUSED_SIZE = 65536` cap means the reduction dimension must be <= 65536 elements.

---

## 1. RMS Norm

**File:** `src/liger_kernel/ops/rms_norm.py`

### Forward: `_rms_norm_forward_kernel`
- **Block/Tile Size:** `BLOCK_SIZE = calculate_settings(n_cols)` -- next_power_of_2 of hidden dim, up to 65536. Passed as `tl.constexpr`.
- **Program ID Mapping:** `row_idx = tl.program_id(0)` -- one program per row. Grid = `(n_rows,)`.
- **Vector Loads:** Single load of entire row: `tl.load(x_base + col_offsets, mask=mask, other=0)` where `col_offsets = tl.arange(0, BLOCK_SIZE)`.
- **Reduction:** `tl.sum(X_row * X_row, axis=0) / n_cols` -- full reduction across hidden dim in one shot.
- **Elements Per Program:** Exactly `n_cols` elements (the hidden dimension).
- **Fusion:** Fuses square, sum, rsqrt, multiply by weight+offset, and optional casting.
- **Edge Cases:** `mask = col_offsets < n_cols` handles non-power-of-2 hidden dims.

### Forward (Block variant): `_block_rms_norm_forward_kernel`
- Processes `BLOCK_ROW = 16` rows per program instance using 2D tiles `[BLOCK_ROW, BLOCK_SIZE]`.
- **Selection logic:** Used when `BLOCK_SIZE <= 256 AND n_rows >= 4096*8 AND NOT row_mode`. For small hidden dims with many rows, the block variant amortizes kernel launch overhead.
- **Elements Per Program:** `BLOCK_ROW * n_cols = 16 * n_cols`.

### Backward: `_rms_norm_backward_kernel`
- **Program ID Mapping:** Grid = `(sm_count,)` -- one program per SM. `rows_per_program = ceil(n_rows / sm_count)`.
- **Work Pattern:** Each program iterates over `rows_per_program` rows in a loop.
- **Accumulation:** `dW_row` accumulates weight gradients in fp32 across all assigned rows.

---

## 2. Layer Norm

**File:** `src/liger_kernel/ops/layer_norm.py`

### Forward: `_layer_norm_forward_kernel`
- **Block/Tile Size:** `BLOCK_SIZE = calculate_settings(n_cols)`.
- **Program ID Mapping:** `row_idx = tl.program_id(0)`. Grid = `(n_rows,)`.
- **Reduction:** Two reductions: `mean = tl.sum(X_f32, axis=0) / n_cols` and `var = tl.sum(X_centered * X_centered, axis=0) / n_cols`.
- **Fusion:** Fuses mean, variance, rsqrt, normalize, affine (weight*x+bias) into single kernel.
- **Edge Cases:** Uses `tl.where(mask, X_centered, 0.0)` to avoid padding contaminating variance.

---

## 3. Group Norm

**File:** `src/liger_kernel/ops/group_norm.py`

### Forward: `_group_norm_forward_kernel`
- **Block/Tile Size:** `BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(hidden_size))`.
- **Program ID Mapping:** 2D grid: `batch_idx = tl.program_id(0)`, `group_idx = tl.program_id(1)`. Grid = `(batch_size, num_groups)`.
- **Multi-pass:** Uses loop `for i in tl.range(0, hidden_size, BLOCK_SIZE)` -- tiles across the group's hidden dimension when `hidden_size > BLOCK_SIZE`.
- **Reduction:** Two-pass: First pass accumulates sum and squared_sum across all tiles. Second pass normalizes.

---

## 4. Softmax

**File:** `src/liger_kernel/ops/softmax.py`

### Single-block forward: `_softmax_single_block_forward_kernel`
- **Block/Tile Size:** `BLOCK_SIZE = calculate_settings(n_cols)`. Used when `n_cols <= BLOCK_SIZE`.
- **Algorithm:** Standard safe softmax: `m = tl.max(x)`, `e = tl.exp(x - m)`, `d = tl.sum(e)`, `y = e / d`.
- **Cache modifiers:** `.ca` on loads (cache at all levels), `.cs` on stores (cache streaming).

### Multi-block forward: `_softmax_multi_block_forward_kernel`
- **When used:** `n_cols > BLOCK_SIZE` (large vocab/feature dims).
- **Algorithm:** Online softmax (two-pass). First pass: iterate over blocks computing running max and running denominator using the online recurrence. Second pass: compute `exp(x - m) / d`.

---

## 5. Cross Entropy

**File:** `src/liger_kernel/ops/cross_entropy.py`

### `liger_cross_entropy_kernel`
- **Block/Tile Size:** `BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))` where `MAX_FUSED_SIZE = 65536 // 2 = 32768`.
- **Program ID Mapping:** `program_id = tl.program_id(0)`. Grid = `(n_rows,)`. One program per token.
- **Algorithm:** Online softmax fused with cross-entropy loss AND gradient computation.
- **In-place gradient storage:** Overwrites input tensor X with gradients to save memory.
- **num_warps:** Hardcoded to 32 (16 on HIP).
- **Fusion:** Extremely aggressive -- softmax + cross-entropy loss + gradient + label smoothing + z-loss + softcapping + argmax tracking all in one kernel.

---

## 6. Elementwise: SwiGLU / GEGLU

**File:** `src/liger_kernel/ops/swiglu.py`, `src/liger_kernel/ops/geglu.py`

### SwiGLU Forward
- **Block/Tile Size:** `BLOCK_SIZE = calculate_settings(n_cols)`.
- **Program ID Mapping:** `program_id = tl.program_id(0)`. Grid = `(n_rows,)`.
- **Computation:** `c = silu(a) * b` where `silu(x) = x * sigmoid(x)`.
- **Memory bound:** Pure elementwise with 2 reads + 1 write per element.

---

## 7. Fused Add + RMS Norm

**File:** `src/liger_kernel/ops/fused_add_rms_norm.py`

- **Fusion pattern:** `S = X + R` (residual add), then RMS norm on S. Fuses the residual connection with normalization.
- **Loads:** X_row, R_row, W_row (3 vectors). Stores: Y_row, S_row, rstd (2 vectors + scalar).
- **This is the decoder layer pattern:** `hidden_states = residual + hidden_states; residual = hidden_states; hidden_states = rmsnorm(hidden_states)`.

---

## 8. DyT (Dynamic Tanh) -- Unique 2D Tiling

**File:** `src/liger_kernel/ops/dyt.py`

### Forward: `_dyt_fwd_kernel`
- **Block/Tile Size:** `BLOCK_N = min(next_power_of_2(N), 2048)` if N >= 4096, else `min(next_power_of_2(N), 1024)`. Hardcoded num_warps=4.
- **Program ID Mapping:** **2D grid:** `col = tl.program_id(0) * BLOCK_N + arange(0, BLOCK_N)`, `row_id = tl.program_id(1)`. Grid = `(cdiv(N, BLOCK_N), M)`.
- **This is the only kernel that tiles the hidden dim across multiple programs** (instead of having one program handle the entire row).

---

## 9. RoPE (Rotary Position Embedding)

**File:** `src/liger_kernel/ops/rope.py`

- **Block/Tile Size:** `BLOCK_SIZE = max(pad_n_q_head, pad_n_kv_head)` -- about heads, not hidden dim.
- **Program ID Mapping:** `pid = tl.program_id(0)`. Grid = `(batch_size * seq_len,)`.
- **2D tile loads:** `[pad_n_qh, pad_hd/2]` tile.
- **Algorithm:** Loads left half and right half, applies rotation: `new_q1 = q1 * cos - q2 * sin; new_q2 = q2 * cos + q1 * sin`.

---

## Summary Table

| Kernel | BLOCK_SIZE Source | Grid Mapping | Multi-pass? | num_warps |
|--------|------------------|--------------|-------------|-----------|
| RMS Norm fwd | `calculate_settings(n_cols)` | 1D: 1 prog/row | No | 4/8/16/32 |
| RMS Norm fwd (block) | same | 1D: BLOCK_ROW=16 rows/prog | No | same |
| RMS Norm bwd | same | 1D: sm_count progs | rows loop | same |
| Layer Norm fwd | `calculate_settings(n_cols)` | 1D: 1 prog/row | No | 4/8/16/32 |
| Group Norm fwd | `min(65536, next_pow2(hidden))` | 2D: (batch, groups) | **Yes** (tiles) | default |
| Softmax fwd (single) | `calculate_settings(n_cols)` | 1D: 1 prog/row | No | 4/8/16/32 |
| Softmax fwd (multi) | same | 1D: 1 prog/row | **Yes** (online) | same |
| Cross Entropy | `min(32768, next_pow2(V))` | 1D: 1 prog/token | **Yes** | 32 |
| SwiGLU fwd | `calculate_settings(n_cols)` | 1D: 1 prog/row | No | 4/8/16/32 |
| DyT fwd | `min(next_pow2(N), 2048)` | **2D: (col_tiles, rows)** | No | 4 |

## Key Patterns for IREE

1. **The dominant pattern is "one program per row, BLOCK_SIZE = entire reduction dim."** Covers RMS norm, Layer norm, softmax (small), SwiGLU, GEGLU, fused-add-rms, PolyNorm, Sparsemax. BLOCK_SIZE equals `next_power_of_2(hidden_dim)`, capped at 65536.

2. **For large dimensions (vocab size), multi-pass tiling is used.** Cross entropy (32768 cap), JSD (65536 cap), KL div (16384 cap). The pattern is `for i in range(0, n_cols, BLOCK_SIZE)` with online accumulation.

3. **Backward kernels use a "persistent thread" pattern:** Grid = `(sm_count,)` with each program iterating over `rows_per_program = ceil(n_rows / sm_count)` rows.

4. **Block variant for small hidden dims:** RMS norm has a special `BLOCK_ROW = 16` path when `BLOCK_SIZE <= 256 AND n_rows >= 32768`.

5. **No autotune is actually used.** DyT and GRPO have commented-out `@triton.autotune`. Everything uses hand-tuned heuristics.

6. **The num_warps heuristic scales with BLOCK_SIZE:** 4 (default) -> 8 (>=2048) -> 16 (>=8192) -> 32 (>=32768).

7. **MAX_FUSED_SIZE varies per kernel type:** 65536 (general), 32768 (cross entropy), 16384 (KL div, TVD), 2048 (DyT). Reflects register pressure / spilling tradeoff.
