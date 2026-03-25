// 4D transpose: BSNH -> BNSH (batch, seq, num_heads, head_dim) -> (batch, num_heads, seq, head_dim)
// tensor<4x2048x32x128xf16> -> tensor<4x32x2048x128xf16>
// This is THE most common transpose in transformer attention: rearranging Q/K/V
// from (batch, seq, heads, headdim) to (batch, heads, seq, headdim) for batched matmul.
//
// Source pattern: Every attention implementation does this. PyTorch permute(0,2,1,3),
// FlashInfer, ThunderKittens. The key insight is that the innermost dim (head_dim=128)
// is NOT transposed -- only the middle dims (seq, heads) swap.
//
// == Optimal Configuration ==
// Block size: 256 threads
// Key insight: d3 (head_dim=128) is contiguous in BOTH input and output.
// So this is NOT a coalescing-conflicting transpose! It's a "batch rearrangement".
// Thread mapping: threads along d3 (128 elements), with vec_width=8 for f16.
//   128 / 8 = 16 threads cover d3, leaving 256/16 = 16 threads for d2 tiling.
// Grid: 4 * 32 * ceil(2048/16) = 4 * 32 * 128 = 16384 workgroups
// Each workgroup copies a 16x128 tile (16 seq positions, all 128 head_dim)
// for one (batch, head) pair.
// Reads: coalesced along d3 (contiguous). Writes: coalesced along d3 (still contiguous).
// NO shared memory needed -- both are coalesced because innermost dim is preserved.

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @transpose_4d_bsnh_to_bnsh() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x2048x32x128xf16>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x32x2048x128xf16>>
  %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [4, 2048, 32, 128], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x2048x32x128xf16>> -> tensor<4x2048x32x128xf16>
  %3 = tensor.empty() : tensor<4x32x2048x128xf16>
  // d0=batch, d1=seq, d2=heads, d3=headdim
  // input: (d0, d1, d2, d3), output: (d0, d2, d1, d3)
  %4 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    ins(%2 : tensor<4x2048x32x128xf16>) outs(%3 : tensor<4x32x2048x128xf16>) {
  ^bb0(%in: f16, %out: f16):
    linalg.yield %in : f16
  } -> tensor<4x32x2048x128xf16>
  iree_tensor_ext.dispatch.tensor.store %4, %1, offsets = [0, 0, 0, 0], sizes = [4, 32, 2048, 128], strides = [1, 1, 1, 1] : tensor<4x32x2048x128xf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x32x2048x128xf16>>
  return
}
