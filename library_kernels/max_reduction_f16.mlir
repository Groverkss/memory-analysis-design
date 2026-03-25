// max_reduction_f16.mlir
// Max reduction along d1: tensor<4096x8192xf16> -> tensor<4096xf16>
// Pattern: First stage of softmax. Uses arith.maximumf (IEEE 754 max, propagates NaN).
//
// == Optimal Configuration ==
// Block size: 256 threads (1D).
// Grid: 4096 blocks (one per row).
// Vector width: 8 (f16 vec8 = 16 bytes, fills 128-bit load).
// Elements per thread: 8192 / 256 = 32 elements per thread.
//   With vec=8: 32 / 8 = 4 vectorized iterations per thread.
//   CUB MemBoundScaling: items_per_thread = 32 * 4 / sizeof(f16) = 64.
//     But actual items_per_thread is capped at 32 by the problem size.
// Reduction strategy: Warp shuffle reduction.
//   Each thread loops over its 32 elements, maintaining a local max.
//   arith.maximumf handles NaN propagation (NaN > anything = NaN).
//   Warp shuffle: __shfl_down with maximumf. 256/32 = 8 warps.
//   8 warp maxima reduced via smem (8 * 2 bytes = 16 bytes smem).
// Coalescing: thread k reads [row, k], [row, k+256], ... -> perfectly coalesced.
//   32 threads in a warp read 32 consecutive f16 values = 64 bytes per transaction.
// Note: This is identical to sum reduction structurally, but uses maximumf instead
//   of addf. The identity element is -inf (0xFC00 for f16).
//   maximumf is a single instruction on modern GPUs (hmax on NVIDIA, v_max_f16 on AMD).

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @max_reduction_f16() {
  %cst_neg_inf = arith.constant 0xFC00 : f16
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x8192xf16>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4096xf16>>
  %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [4096, 8192], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x8192xf16>> -> tensor<4096x8192xf16>
  %3 = tensor.empty() : tensor<4096xf16>
  %4 = linalg.fill ins(%cst_neg_inf : f16) outs(%3 : tensor<4096xf16>) -> tensor<4096xf16>
  %5 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]}
    ins(%2 : tensor<4096x8192xf16>) outs(%4 : tensor<4096xf16>) {
  ^bb0(%in: f16, %out: f16):
    %6 = arith.maximumf %in, %out : f16
    linalg.yield %6 : f16
  } -> tensor<4096xf16>
  iree_tensor_ext.dispatch.tensor.store %5, %1, offsets = [0], sizes = [4096], strides = [1] : tensor<4096xf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4096xf16>>
  return
}
