# FlashInfer: Memory-Bound Kernel Analysis

---

## Exhaustive Analysis of Memory-Bound Triton Kernels in Unsloth

### Key Utility: `calculate_settings()` (used by most kernels)

**File**: `/home/kunwar/Work/kernel_libs/unsloth/unsloth/kernels/utils.py`, lines 100-119

```python
def calculate_settings(n: int) -> (int, int):
    BLOCK_SIZE = next_power_of_2(n)  # rounds up to power of 2
    # capped at MAX_FUSED_SIZE = 65536
    num_warps = 4
    if BLOCK_SIZE >= 32768: num_warps = 32
    elif BLOCK_SIZE >= 8192: num_warps = 16
    elif BLOCK_SIZE >= 2048: num_warps = 8
```

This is the single shared heuristic for BLOCK_SIZE and num_warps. It always rounds `n_cols` up to the next power of 2. There is no autotuning. `MAX_FUSED_SIZE = 65536` (line 19).

---

### 1. RMS LayerNorm Forward

**File**: `/home/kunwar/Work/kernel_libs/unsloth/unsloth/kernels/rms_layernorm.py`, lines 21-59

#### Block/Tile Sizes
- `BLOCK_SIZE = next_power_of_2(n_cols)` via `calculate_settings`. One tile covers the **entire row** (hidden dim). No loop over columns.
- For typical hidden dims: dim=2048 -> BLOCK_SIZE=2048, num_warps=8; dim=4096 -> BLOCK_SIZE=4096, num_warps=8; dim=8192 -> BLOCK_SIZE=8192, num_warps=16.

#### Program ID Mapping
- **1D grid**: `(n_rows,)` where `n_rows = batch_size * seq_len` (input is reshaped to `[-1, dim]`).
- `row_idx = tl.program_id(0)` -- one program instance per row.

#### Vector Loads
- `col_offsets = tl.arange(0, BLOCK_SIZE)` -- contiguous vector of [0, BLOCK_SIZE).
- `mask = col_offsets < n_cols` -- handles non-power-of-2 hidden dims.
- `X_row = tl.load(X + col_offsets, mask=mask, other=0).to(tl.float32)` -- **single vector load of entire row**, cast to f32.
- `W_row = tl.load(W + col_offsets, mask=mask, other=0)` -- weight loaded in original dtype (no `.to(tl.float32)`).
- Pointer arithmetic: `X += row_idx * X_row_stride` advances to the row start. Then `X + col_offsets` gives contiguous memory access (coalesced).

#### Reduction Strategy
- `row_var = tl.sum(X_row * X_row, axis=0) / n_cols` -- single `tl.sum` reduction for variance, no mean subtraction (RMS).
- `inv_var = tl.math.rsqrt(row_var + eps_f32)` -- reciprocal sqrt.

#### Fusion Patterns
- Fuses: (1) load X, (2) compute variance, (3) rsqrt, (4) normalize, (5) scale by W, (6) store Y. All in one kernel.
- Also stores `inv_var` to `r` for backward pass reuse.

#### Memory Access Patterns
- Row-major layout. Each program loads one contiguous row of X and one row of W (broadcast across rows).
- `Y_row_stride`, `X_row_stride`, `W_row_stride` passed as `tl.constexpr` so Triton can optimize pointer arithmetic at compile time.
- W is shared across all rows (broadcast), so it benefits from L2 cache after first row loads it.

#### Elements Per Program
- Exactly `n_cols` elements of X, `n_cols` elements of W, writes `n_cols` elements of Y + 1 scalar `inv_var`.

#### Edge Cases
- `mask = col_offsets < n_cols` handles non-power-of-2 dims. `other=0` ensures masked elements contribute 0 to the sum.

---

### 2. Gemma RMS LayerNorm Forward

**File**: `/home/kunwar/Work/kernel_libs/unsloth/unsloth/kernels/rms_layernorm.py`, lines 123-159

Identical structure to the standard RMS LayerNorm forward with one difference:

```python
output = normed * (W_row + 1.0)  # line 157
```

Instead of `normed * W_row`. This is Gemma's `(1 + weight)` convention. W_row is also loaded as `.to(tl.float32)` (line 149), unlike the standard kernel where W stays in original dtype.

---

### 3. RMS LayerNorm Backward

**File**: `/home/kunwar/Work/kernel_libs/unsloth/unsloth/kernels/rms_layernorm.py`, lines 62-120

#### Block/Tile Sizes
- Same as forward: `BLOCK_SIZE` from `calculate_settings(n_cols)`, reused from forward context.

#### Program ID Mapping
- 1D grid `(n_rows,)`, `row_idx = tl.program_id(0)`.

#### Vector Loads
- Loads 3 full rows: `dY_row`, `X_row`, `W_row` -- all cast to f32.
- Loads scalar `inv_var = tl.load(r)`.

#### Reduction Strategy
- `rowsum_dY_normed = tl.sum(dY_W * normed, axis=0)` -- single reduction.

#### Fusion Patterns
- Fuses entire backward: load dY, X, W, r -> compute normed -> compute gradient -> store dX.
- Has a `GEMMA` constexpr heuristic that switches between `dY_W = dY_row * W_row` and `dY_W = dY_row * (W_row + 1.0)`.

#### Edge Case
- When `GEMMA=False`: `dX = dY` (in-place write to dY buffer, line 95). When `GEMMA=True`: allocates separate `dX`.
- Applied via `triton.heuristics` decorator (lines 116-120).

---

### 4. LayerNorm Forward

**File**: `/home/kunwar/Work/kernel_libs/unsloth/unsloth/kernels/layernorm.py`, lines 25-64

#### Block/Tile Sizes
- Same `calculate_settings(n_cols)` pattern.

#### Program ID Mapping
- 1D grid `(n_rows,)`. `row_idx = tl.program_id(0)`.

#### Vector Loads
- Loads X_row, W_row, b_row -- all cast `.to(tl.float32)`.
- Mask pattern identical to RMS norm.

#### Reduction Strategy
- **Two reductions** (vs. one for RMS):
  1. `mean_X = tl.sum(X_row, axis=0) / n_cols` -- mean.
  2. `XX = tl.where(mask, X_row - mean_X, 0)` then `row_var = tl.sum(XX * XX, axis=0) / n_cols` -- variance.
- Note the `tl.where(mask, X_row - mean_X, 0)` at line 56 to avoid `(0 - mean)` contributing to variance from masked positions.

#### Fusion Patterns
- Fuses: load X -> mean -> variance -> rsqrt -> normalize -> affine (W, b) -> store Y.
- Stores both `inv_var` and `mean_X` for backward.

#### Memory Access
- `r += row_idx` and `mu += row_idx` -- scalar per-row storage, stride=1.

---

### 5. LayerNorm Backward

**File**: `/home/kunwar/Work/kernel_libs/unsloth/unsloth/kernels/layernorm.py`, lines 67-108

#### Reduction Strategy
- Two reductions in the gradient formula:
  1. `tl.sum(dY_W, axis=0)` -- sum of grad-weight product.
  2. `tl.sum(dY_W * normed, axis=0)` -- sum of grad-weight-normed product.
- Implements the Karpathy llm.c backward formula.

#### Fusion
- Writes in-place to `dY` buffer (line 108): `tl.store(dY + col_offsets, dX_row, mask=mask)`.

---

### 6. Cross-Entropy Forward (Small Vocab)

**File**: `/home/kunwar/Work/kernel_libs/unsloth/unsloth/kernels/cross_entropy_loss.py`, lines 35-113

#### Block/Tile Sizes
- For vocab <= 65536: `BLOCK_SIZE = next_power_of_2(vocab_size)` via `calculate_settings`.
- On CDNA (AMD MI): `num_warps = num_warps // 2` (line 319).

#### Program ID Mapping
- 1D grid `(n_rows,)` where `n_rows = batch * seq_len`.
- `row_idx = tl.program_id(0)`.

#### Vector Loads
- `col_offsets = tl.arange(0, BLOCK_SIZE)`, `mask = col_offsets < VOCAB_SIZE`.
- Full vocab row loaded in one shot: `tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf"))` -- uses `-inf` for out-of-range so they contribute 0 to softmax.
- `label_idx = tl.load(labels_ptr).to(tl.int32)` -- scalar.

#### Reduction Strategy
- `c = tl.max(logits, 0)` -- max for numerical stability.
- `logsumexp = c + tl.log(tl.sum(tl.exp(logits - c), 0))` -- fused logsumexp.

#### Fusion Patterns
- Optional logit scaling (Cohere): `logits = LOGIT_SCALE * logits`.
- Optional softcapping (Gemma 2): `logits = SOFTCAP * tanh(logits / SOFTCAP)`.
- Controlled via `triton.heuristics` on `DO_SOFTCAPPING` and `DO_LOGIT_SCALING` (lines 108-113) -- eliminated at compile time when not needed.
- Label `x` is re-loaded individually (line 93): `x = tl.load(logits_ptr + label_idx)` -- scalar load for the target logit.
- Loss = `logsumexp - x` or 0 if `label_idx == -100` (ignore index).

#### Pointer Arithmetic
- `logits_ptr += row_idx * triton_cast(logits_row_stride, tl.int64)` -- cast to int64 for large tensors.

#### Elements Per Program
- Loads `VOCAB_SIZE` logits, 1 label. Stores 1 loss + 1 logsumexp scalar.

---

### 7. Cross-Entropy Forward (Chunked, Large Vocab)

**File**: `/home/kunwar/Work/kernel_libs/unsloth/unsloth/kernels/cross_entropy_loss.py`, lines 116-203

For vocabs > 65536 (e.g., Gemma 256K).

#### Block/Tile Sizes
- `BLOCK_SIZE = MAX_FUSED_SIZE = 65536` (hardcoded).
- `N_CHUNKS = ceil(vocab_size / 65536)`.
- `num_warps = 32` (or 16 on CDNA).

#### Program ID Mapping
- **2D grid**: `(n_rows, n_chunks)`.
- `row_idx = tl.program_id(0)`, `chunk_idx = tl.program_id(1)`.

#### Vector Loads
- `col_offsets = chunk_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)` -- each chunk handles a different 65536-element slice.
- `mask = col_offsets < VOCAB_SIZE` -- only the last chunk may be partially masked.

#### Reduction Strategy
- Per-chunk logsumexp: `c = tl.max(logits, 0)`, `logsumexp = c + tl.log(tl.sum(tl.exp(logits - c), 0))`.
- **Final reduction done on host** via `torch.logsumexp(logsumexp, dim=1)` (line 371).
- Only `chunk_idx == 0` computes the `-x` part of the loss (lines 179-193).

#### Elements Per Program
- 65536 logits per chunk program.

---

### 8. Cross-Entropy Backward

**File**: `/home/kunwar/Work/kernel_libs/unsloth/unsloth/kernels/cross_entropy_loss.py`, lines 206-289

#### Block/Tile Sizes
- `BLOCK_SIZE = 4096` (hardcoded constant, line 389).
- `n_blocks = ceil(vocab_size / 4096)`.
- `num_warps = 8` (line 414).

#### Program ID Mapping
- 2D grid: `(n_rows, n_blocks)`.
- `row_idx = tl.program_id(0)`, `block_idx = tl.program_id(1)`.

#### Vector Loads
- `col_offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)`.
- Loads logits, logsumexp (scalar), labels (scalar), dloss (scalar).

#### Computation
- `y = tl.exp(x - logsumexp)` -- softmax.
- `y = tl.where(col_offsets == label_idx, y - 1.0, y)` -- subtract 1 at label position.
- Writes **in-place** to `logits_ptr` (line 280): `tl.store(logits_ptr + col_offsets, dloss * y, mask=mask)`.

#### Fusion
- Includes softcapping backward: `y = y * (1.0 - partial * partial)` (sech^2 = 1 - tanh^2).
- Includes logit scaling backward: `y = y * LOGIT_SCALE`.

---

### 9. SwiGLU Forward (`_fg_kernel`)

**File**: `/home/kunwar/Work/kernel_libs/unsloth/unsloth/kernels/swiglu.py`, lines 27-57

#### Block/Tile Sizes
- `BLOCK_SIZE = 1024` (module-level constant, line 23).
- Grid: `triton.cdiv(n_elements, BLOCK_SIZE)` -- covers flattened tensor.

#### Program ID Mapping
- 1D grid over **total elements** (not rows).
- `block_idx = tl.program_id(0)`.
- `offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)`.

#### Vector Loads
- Flat 1D addressing. Loads `e` (gate) and `g` (up) at same offsets.
- `e_row = tl.load(e + offsets, mask=mask, other=0).to(tl.float32)` -- gate in f32.
- `g_row = tl.load(g + offsets, mask=mask, other=0)` -- up stays in original dtype.

#### Reduction Strategy
- **None**. This is purely elementwise.

#### Fusion Pattern
- Fuses: `SiLU(gate) * up` = `(gate * sigmoid(gate)) * up`.
- `f_row = e_row * tl.sigmoid(e_row)` then cast back to original dtype, then `h_row = f_row * g_row`.

#### Long Indexing
- `LONG_INDEXING` flag: when `n_elements > INT32_SAFETY_BUFFER (2^31 - 4096)`, offsets are computed in int64 (lines 37-43). This handles very large tensors beyond int32 range.

#### Elements Per Program
- 1024 elements per program instance.

---

### 10. SwiGLU Backward (`_DWf_DW_dfg_kernel`)

**File**: `/home/kunwar/Work/kernel_libs/unsloth/unsloth/kernels/swiglu.py`, lines 76-128

#### Block/Tile Sizes
- Same `BLOCK_SIZE = 1024`.

#### Fusion Pattern
- Fuses the entire backward into one kernel computing 3 outputs:
  - `h = f * g` (recomputed forward output, stored to DW buffer)
  - `df = DW * f` (stored to e buffer)
  - `de = dg * sigmoid(e) * (1 + e * (1 - sigmoid(e)))` (stored to g buffer)
- **Reuses input buffers for output** -- writes `h_row` to DW, `df_row` to e, `de_row` to g.

#### Memory Access
- Same flat 1D pattern. 3 loads, 3 stores per element.

---

### 11. GEGLU Exact Forward (`_exact_forward_kernel`)

**File**: `/home/kunwar/Work/kernel_libs/unsloth/unsloth/kernels/geglu.py`, lines 31-61

#### Block/Tile Sizes
- `BLOCK_SIZE = 1024` (module-level constant, line 27).

#### Fusion Pattern
- Fuses: `GELU(gate) * up` where GELU is exact (erf-based).
- `f_row = 0.5 * e_row * (tl.math.erf(tl.math.rsqrt(2.0) * e_row) + 1.0)` (line 55).

#### Everything else identical to SwiGLU forward pattern (1D flat, mask, LONG_INDEXING).

---

### 12. GEGLU Exact Backward (`_exact_backward_kernel`)

**File**: `/home/kunwar/Work/kernel_libs/unsloth/unsloth/kernels/geglu.py`, lines 81-138

#### Fusion Pattern
- Computes derivative: `df/de = 0.5 * (1 + erf(x/sqrt(2))) + (1/sqrt(2*pi)) * x * exp(-0.5*x^2)`.
- Constant `t = 0.3989422804014327` = `1/sqrt(2*pi)` (line 128).
- Same buffer-reuse pattern: writes h_row to DW, df_row to e, de_row to g.

---

### 13. GEGLU Approximate Forward (`_approx_forward_kernel`)

**File**: `/home/kunwar/Work/kernel_libs/unsloth/unsloth/kernels/geglu.py`, lines 156-191

#### Fusion Pattern
- Uses tanh approximation: `f = 0.5 * x * (1 + tanh(sqrt(2/pi) * x * (1 + 0.044715 * x^2)))`.
- Constant `s = 0.7978845608028654` = `sqrt(2/pi)` (line 178).
- Uses `triton_tanh` which resolves to `libdevice.tanh` on Triton 3.0+ (from utils.py).

---

### 14. GEGLU Approximate Backward (`_approx_backward_kernel`)

**File**: `/home/kunwar/Work/kernel_libs/unsloth/unsloth/kernels/geglu.py`, lines 211-274

#### Fusion Pattern
- Derivative via the identity `sech^2(x) = 1 - tanh^2(x)` to reuse tanh computation.
- Formula: `df/de = 0.5*(1+tanh(...)) + 0.5*(1-tanh(...)^2)*(sqrt(2/pi)*x*(1+3*0.044715*x^2))`.
- Refactored as `T = 1 + tanh(a+b)`, `Q2 = -T2*(T-2)*(a+3b)`, `df_de = T2 + Q2`.

---

### 15. RoPE Embedding (Single Q/K, `_rope_embedding`)

**File**: `/home/kunwar/Work/kernel_libs/unsloth/unsloth/kernels/rope_embedding.py`, lines 112-182

#### Block/Tile Sizes
- `BLOCK_SIZE = next_power_of_2(head_dim // 2)` via `calculate_settings` (line 202).
- Typical: head_dim=128 -> BLOCK_SIZE=64, num_warps=4.

#### Program ID Mapping
- **2D grid**: `(n_rows, n_groups)` where `n_rows = batch * seq_len`, `n_groups = ceil(n_heads / ROPE_GROUP_SIZE)`.
- `ROPE_GROUP_SIZE = 4` (line 109). Each program handles 4 heads.
- `row_position = tl.program_id(0)`, `group_head_position = tl.program_id(1)`.

#### Vector Loads
- `col_offsets = tl.arange(0, BLOCK_SIZE)`, `mask = col_offsets < half_head_dim`.
- Loads sin/cos once per program: `sin1 = tl.load(sin + (row_position % seqlen) * sin_row_stride + col_offsets, mask=mask)`.
- Then loops over 4 heads (lines 163-174):
  ```python
  for k in range(head_start, head_end):
      offs_q1 = row_position * Q_row_stride + k * head_dim + col_offsets
      offs_q2 = row_position * Q_row_stride + k * head_dim + col_offsets + half_head_dim
      Q1 = tl.load(Q + offs_q1, mask=mask)
      Q2 = tl.load(Q + offs_q2, mask=mask)
      tl.store(Q + offs_q1, Q1*cos1 - Q2*sin1, mask=mask)
      tl.store(Q + offs_q2, Q2*cos1 + Q1*sin1, mask=mask)
  ```

#### Fusion Pattern
- Fuses: (1) load sin/cos, (2) load Q first-half + second-half, (3) rotation, (4) store. All in-place.
- Sin/cos loaded once, reused across 4 heads.
- For backward, `sin1 = -sin1` (line 156) -- same kernel.

#### Memory Access Pattern
- Each head's data is `head_dim` contiguous elements. Loads two halves separately (first half, second half).
- Pointer arithmetic: `row_position * Q_row_stride + k * head_dim + col_offsets` -- not fully coalesced across threads since different threads in a warp access different positions within the same head.

#### Elements Per Program
- `4 * 2 * half_head_dim` reads + `4 * 2 * half_head_dim` writes = `4 * head_dim` reads and writes of Q.
- Plus `half_head_dim` each for sin and cos.

---

### 16. RoPE Embedding QK (`_rope_embedding_QK`)

**File**: `/home/kunwar/Work/kernel_libs/unsloth/unsloth/kernels/rope_embedding.py`, lines 23-106

#### Block/Tile Sizes
- `BLOCK_SIZE = next_power_of_2(head_dim)` via `calculate_settings` (line 324). Note: uses full `head_dim`, not `head_dim//2`.

#### Program ID Mapping
- 2D grid: `(batch * seq_len, n_heads_Q)`.
- `row_position = tl.program_id(0)`, `head_position = tl.program_id(1)`.
- One program per (position, head) pair.

#### Vector Loads
- `mask = col_offsets < half_head_dim` (half of BLOCK_SIZE used).
- Q loads: `q0 = tl.load(q_ptr + col_offsets)`, `q1 = tl.load(q_ptr + half_head_dim + col_offsets)`.
- K loads (conditional): only if `head_position < n_heads_K` (GQA support).

#### Pointer Arithmetic
- Uses separate batch/head/seq strides for full layout flexibility:
  ```python
  q_ptr = Q + batch_id * Q_batch_stride + head_position * Q_head_stride + seq_index * Q_seq_stride
  ```

#### Fusion
- Fuses Q and K rotation in the same kernel. K is only processed when `head_position < n_heads_K`.
- Optional rope_embedding_indices for position remapping (e.g., TRL).
- `HAS_ROPE_INDICES` heuristic controls compile-time branch elimination.

#### Elements Per Program
- 2 * half_head_dim for Q + conditionally 2 * half_head_dim for K.

---

### 17. FP8 Weight Dequant Kernel (`weight_dequant_kernel`)

**File**: `/home/kunwar/Work/kernel_libs/unsloth/unsloth/kernels/fp8.py`, lines 64-76

#### Block/Tile Sizes
- `BLOCK_SIZE = block_size` parameter (default 128 from caller, line 93).

#### Program ID Mapping
- 2D grid: `(ceil(M/BLOCK_SIZE), ceil(N/BLOCK_SIZE))`.
- `pid_m = tl.program_id(0)`, `pid_n = tl.program_id(1)`.

#### Vector Loads
- 2D tile: `offs_m[:, None] * N + offs_n[None, :]` -- BLOCK_SIZE x BLOCK_SIZE tile.
- Mask: `(offs_m[:, None] < M) & (offs_n[None, :] < N)`.
- Scale is per-block: `s = tl.load(s_ptr + pid_m * n + pid_n)` -- single scalar per tile.

#### Fusion
- Simple dequant: `y = x.to(f32) * scale`.

Not memory-bound in the traditional sense (this is a compute-trivial dequant), but included for completeness.

---

### 18. FP8 Activation Quant Kernel (`act_quant_kernel`)

**File**: `/home/kunwar/Work/kernel_libs/unsloth/unsloth/kernels/fp8.py`, lines 118-131

#### Block/Tile Sizes
- `BLOCK_SIZE = 128` (from caller, line 146).

#### Program ID Mapping
- 1D grid: `ceil(numel / BLOCK_SIZE)`.
- `pid = tl.program_id(0)`.

#### Vector Loads
- `offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)`.
- **No mask** -- requires `x.shape[-1] % block_size == 0` (asserted at line 139).

#### Reduction
- `s = tl.max(tl.abs(x)) / 448.0` -- per-block max for FP8 scale.
- Special case: `s = 1.0 if s == 0 else s` (line 127) to avoid NaN from zero rows.

#### Fusion
- Fuses: abs-max -> scale computation -> quantize -> store quantized + scale.

---

## Summary Table

| Kernel | File | Lines | BLOCK_SIZE | Grid | Reduction | Fusion |
|--------|------|-------|------------|------|-----------|--------|
| RMS Norm Fwd | rms_layernorm.py | 21-59 | next_pow2(n_cols) | (n_rows,) | tl.sum (variance) | load+norm+scale+store |
| Gemma RMS Fwd | rms_layernorm.py | 123-159 | next_pow2(n_cols) | (n_rows,) | tl.sum (variance) | load+norm+scale(+1)+store |
| RMS Norm Bwd | rms_layernorm.py | 62-120 | next_pow2(n_cols) | (n_rows,) | tl.sum (grad) | full backward fused |
| LayerNorm Fwd | layernorm.py | 25-64 | next_pow2(n_cols) | (n_rows,) | 2x tl.sum (mean+var) | load+mean+var+norm+affine+store |
| LayerNorm Bwd | layernorm.py | 67-108 | next_pow2(n_cols) | (n_rows,) | 2x tl.sum | full backward fused |
| CE Fwd (small) | cross_entropy_loss.py | 35-113 | next_pow2(vocab) | (n_rows,) | tl.max + tl.sum | logsumexp+loss+softcap+scaling |
| CE Fwd (chunk) | cross_entropy_loss.py | 116-203 | 65536 fixed | (n_rows, n_chunks) | tl.max + tl.sum per chunk | partial logsumexp per chunk |
| CE Bwd | cross_entropy_loss.py | 206-289 | 4096 fixed | (n_rows, n_blocks) | None | softmax-grad+softcap+scaling |
| SwiGLU Fwd | swiglu.py | 27-57 | 1024 fixed | (n_elements/1024,) | None (elementwise) | silu(gate)*up |
| SwiGLU Bwd | swiglu.py | 76-128 | 1024 fixed | (n_elements/1024,) | None | full backward, buffer reuse |
| GEGLU Exact Fwd | geglu.py | 31-61 | 1024 fixed | (n_elements/1024,) | None | gelu_exact(gate)*up |
| GEGLU Exact Bwd | geglu.py | 81-138 | 1024 fixed | (n_elements/1024,) | None | full backward, buffer reuse |
| GEGLU Approx Fwd | geglu.py | 156-191 | 1024 fixed | (n_elements/1024,) | None | gelu_tanh(gate)*up |
| GEGLU Approx Bwd | geglu.py | 211-274 | 1024 fixed | (n_elements/1024,) | None | full backward, buffer reuse |
| RoPE (single) | rope_embedding.py | 112-182 | next_pow2(hd/2) | (n_rows, n_groups) | None | sin/cos rotation, 4 heads |
| RoPE QK | rope_embedding.py | 23-106 | next_pow2(hd) | (batch*seq, n_heads_Q) | None | Q+K rotation fused |
| FP8 Dequant | fp8.py | 64-76 | 128 | 2D (M/bs, N/bs) | None | x*scale |
| FP8 Act Quant | fp8.py | 118-131 | 128 | (numel/128,) | tl.max(abs) | absmax+quantize |

### Key Design Patterns Across All Kernels

1. **No autotuning anywhere**. All BLOCK_SIZE choices are either a fixed constant (1024 for elementwise, 4096/65536 for CE) or `next_power_of_2(dim)` via `calculate_settings`.

2. **Norm kernels process entire rows in a single program** -- no loop over columns. This limits them to hidden dims <= 65536.

3. **Elementwise kernels (SwiGLU, GEGLU) use flat 1D addressing** with a fixed BLOCK_SIZE=1024, independent of tensor shape.

4. **Buffer reuse in backward passes** -- SwiGLU and GEGLU backwards write outputs into input buffers (DW, e, g), avoiding extra allocations.

5. **`tl.constexpr` on strides** -- row strides are compile-time constants, enabling Triton to optimize pointer arithmetic.

6. **int64 safety** -- elementwise kernels have `LONG_INDEXING` flag for tensors with >2^31 elements. Cross-entropy uses `triton_cast(stride, tl.int64)`.

7. **All masking uses `other=0`** (contributing nothing to sums) except cross-entropy which uses `other=-float("inf")` (contributing nothing to softmax).