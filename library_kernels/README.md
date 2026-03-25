# Library Kernel Test Bench

52 MLIR kernel test cases representing memory-bound patterns found across 12 kernel
libraries (Apex, PyTorch, llama.cpp, MIOpen, Liger-Kernel, Unsloth, FlashInfer, Quack,
ThunderKittens, CUTLASS, CUB/CCCL, TVM).

Each kernel includes an "== Optimal Configuration ==" comment block describing what
a good GPU configuration should look like, based on the cross-cutting analysis in
`kernel_analysis/00_cross_cutting_lessons.md`.

## Kernel Categories

### Normalization (12 kernels)

| File | Shape | Type | Key Characteristic |
|------|-------|------|--------------------|
| `rms_norm_small_hidden.mlir` | 8192x128 | f16->f32 | Small hidden, multiple rows/block |
| `rms_norm_medium_hidden.mlir` | 2048x4096 | f16->f32 | Standard LLM (Llama/GPT) |
| `rms_norm_large_hidden.mlir` | 512x16384 | bf16->f32 | Large model, all threads on one row |
| `rms_norm_unaligned.mlir` | 4096x768 | f16->f32 | BERT-base, non-power-of-2 hidden |
| `rms_norm_dynamic.mlir` | ?x? | f16->f32 | Dynamic batch + hidden dims |
| `rms_norm_very_large.mlir` | 64x65536 | f16->f32 | Needs multi-block per row |
| `layernorm_standard.mlir` | 2048x4096 | f16->f32 | Full layernorm (mean+var+normalize) |
| `layernorm_small_hidden.mlir` | 32768x64 | f32 | Per-head norm, single warp |
| `layernorm_3d.mlir` | 8x512x10x128 | f16 | Multi-dim reduction (d2,d3) |
| `layernorm_unaligned_3d.mlir` | 16x100x511 | f16 | Prime reduction dim (511) |
| `groupnorm_nhwc.mlir` | 4x64x32x32 | f32 | 8 groups, spatial+channel reduce |
| `fused_residual_rms_norm.mlir` | 2048x4096 | f16 | Decoder-layer fusion: add+norm |

### Softmax (8 kernels)

| File | Shape | Key Characteristic |
|------|-------|--------------------|
| `softmax_small.mlir` | 512x128 f32 | Warp-level, fits in registers |
| `softmax_medium.mlir` | 2048x4096 f16 | Standard attention softmax |
| `softmax_large_vocab.mlir` | 4096x32000 f16 | Vocab-size, multi-pass |
| `softmax_very_large_vocab.mlir` | 2048x128256 f16 | Gemma vocab, 16+ passes |
| `softmax_outer_dim.mlir` | 4096x1024 f32 | Column-wise (outer reduction) |
| `softmax_3d.mlir` | 8x16x2048 f16 | Multi-head attention |
| `softmax_dynamic.mlir` | ?x? f32 | Dynamic dimensions |
| `softmax_unaligned.mlir` | 1024x511 f16 | Non-power-of-2 feature dim |

### Elementwise / Activations (12 kernels)

| File | Shape | Key Characteristic |
|------|-------|--------------------|
| `silu_elementwise.mlir` | 2048x4096 f16 | SiLU = x*sigmoid(x) |
| `swiglu_fused.mlir` | 2048x4096 f16 | silu(gate)*up, 2 inputs |
| `geglu_fused.mlir` | 2048x4096 f16 | gelu_tanh(gate)*up |
| `elementwise_transpose.mlir` | 1024x4096 f16->f32 | Coalescing conflict (transpose) |
| `elementwise_broadcast.mlir` | 2048x4096 + 4096 f32 | Bias add, 1D broadcast |
| `elementwise_large_1d.mlir` | 16777216 f16 | 16M elements, pure 1D |
| `elementwise_dynamic.mlir` | ?x? f32 | Dynamic ReLU |
| `elementwise_unaligned.mlir` | 1000x3 f32 | Tiny innermost dim (3) |
| `rope_embedding.mlir` | 4x2048x32x128 f16 | Rotary position embedding |
| `elementwise_mixed_precision.mlir` | 4096x4096 f8->f16 | FP8 dequantization |
| `elementwise_3d_broadcast.mlir` | 8x512x4096 + 4096 f16 | 3D + 1D broadcast |
| `elementwise_tiny_batch.mlir` | 1x65536 f16 | Single row, very wide |

### Transpose (4 kernels)

| File | Shape | Key Characteristic |
|------|-------|--------------------|
| `transpose_pure_2d.mlir` | 2048x4096 f16 -> 4096x2048 | Classic shared-memory transpose |
| `transpose_3d_inner.mlir` | 64x512x1024 f32 -> 64x1024x512 | Batched K^T for attention |
| `transpose_4d_bsnh_to_bnsh.mlir` | 4x2048x32x128 f16 -> 4x32x2048x128 | QKV reshape, innermost dim preserved (no conflict!) |
| `layernorm_transpose_output.mlir` | 2048x4096 f16 -> 4096x2048 f32 | LayerNorm + transposed write (hardest coalescing case) |

### Outer Reductions (4 additional kernels)

| File | Shape | Key Characteristic |
|------|-------|--------------------|
| `outer_reduction_f16.mlir` | 4096x8192 f16 -> 8192 | Large outer reduce, 2D thread block |
| `outer_reduction_small_parallel.mlir` | 8192x64 f32 -> 64 | BN-style: 1 WG per channel, strided reads |
| `outer_reduction_3d.mlir` | 256x32x4096 f16 -> 32x4096 | BatchNorm: reduce batch, keep channel+spatial |
| `outer_max_reduction.mlir` | 2048x512 f32 -> 512 | Column-wise max, -inf init |

### Transposed Norm/Softmax Variants (3 additional kernels)

| File | Shape | Key Characteristic |
|------|-------|--------------------|
| `rms_norm_transpose_input.mlir` | 4096x2048 f16 | Reduce along d0 (NON-contiguous), outer reduction norm |
| `softmax_transpose_output.mlir` | 1024x2048 f16 -> 2048x1024 | Softmax + transposed write, coalescing conflict on output |
| `outer_reduction_welford.mlir` | 256x1024 f16 -> (1024, 1024) f32 | Welford mean+var (BatchNorm stats), 2 output tensors |

### Reductions / MatVec / Misc (12 kernels)

| File | Shape | Key Characteristic |
|------|-------|--------------------|
| `inner_reduction_f32.mlir` | 8192x256 f32 | Small reduction, many rows |
| `inner_reduction_f16_large.mlir` | 1024x65536 f16 | Very large inner reduction |
| `outer_reduction.mlir` | 4096x1024 f32 | Column-wise sum, strided |
| `outer_reduction_dynamic.mlir` | ?x? f32 | Dynamic outer reduction |
| `global_reduction.mlir` | 16777216 f32 | Reduce to scalar, multi-block |
| `argmax_row.mlir` | 2048x32000 f32 | Carry value+index, vocab sampling |
| `matvec_f16.mlir` | 4096x4096 * 4096 f16 | GEMV, one block per row |
| `matvec_transposed.mlir` | 4096x4096^T * 4096 f16 | Transposed GEMV, outer reduction |
| `row_sum_unaligned.mlir` | 1000x511 f16 | Both dims non-power-of-2 |
| `reduction_3d.mlir` | 8x256x4096 f32 | Middle-dim reduction |
| `max_reduction_f16.mlir` | 4096x8192 f16 | Max (first softmax stage) |
| `cross_entropy_fused.mlir` | 2048x32000 f16 | Fused logsumexp (max+exp+sum+log) |

## Design Axes Covered

### Dimension Sizes
- **Tiny**: 3, 64, 128 (sub-warp)
- **Small**: 256, 511, 768 (1-2 warps)
- **Medium**: 1024, 2048, 4096 (standard LLM)
- **Large**: 8192, 16384, 32000 (large models, vocab)
- **Very Large**: 65536, 128256 (multi-block needed)

### Alignment
- **Power-of-2**: 128, 256, 1024, 4096, 16384, 65536
- **Non-power-of-2**: 3, 100, 511, 768, 1000, 32000, 128256
- **Prime-ish**: 511, 32000 (tests masking)

### Dynamic Shapes
- `rms_norm_dynamic.mlir` -- both dims dynamic
- `softmax_dynamic.mlir` -- both dims dynamic
- `elementwise_dynamic.mlir` -- both dims dynamic
- `outer_reduction_dynamic.mlir` -- both dims dynamic

### Element Types
- f16 (most kernels)
- bf16 (`rms_norm_large_hidden`)
- f32 (softmax small/outer, reductions, elementwise)
- f8E4M3FNUZ (`elementwise_mixed_precision`)
- Mixed precision: f16->f32 (norms, transpose), f8->f16 (dequant)

### Reduction Patterns
- **Inner reduction** (contiguous dim): norms, row softmax, row sum, matvec
- **Outer reduction** (strided dim): column softmax, column sum, transposed matvec
- **Middle-dim reduction**: `reduction_3d.mlir`
- **Multi-dim reduction**: `layernorm_3d.mlir` (reduce d2,d3)
- **Global reduction**: `global_reduction.mlir` (reduce to scalar)
- **Multi-stage reduction**: softmax (max+sum), layernorm (mean+var), cross-entropy (max+exp+sum+log)

### Coalescing Challenges
- **No conflict**: most inner reductions, elementwise
- **Transpose conflict**: `elementwise_transpose.mlir` (input vs output contiguous dims differ)
- **Broadcast**: bias add, norm weight broadcast
- **Strided access**: outer reductions, transposed matvec
- **Tiny innermost**: `elementwise_unaligned.mlir` (dim=3, can't vectorize)

### Multi-Op Fusions
- **Norm fusions**: RMSNorm (extf+mul+reduce+rsqrt+mul+truncf), LayerNorm (mean+var+normalize)
- **Residual fusion**: `fused_residual_rms_norm.mlir` (add + norm)
- **Gate fusions**: SwiGLU (silu+mul), GEGLU (gelu+mul)
- **Softmax**: 4-op chain (max+exp+sum+div)
- **Cross-entropy**: 4-op chain (max+exp+sum+log)
