// row_sum_unaligned.mlir
// Sum reduction: tensor<1000x511xf16> -> tensor<1000xf16>
// Pattern: Non-power-of-2 in both dims. Tests masking and alignment fallback.
// 1000 rows (not power-of-2), 511 columns (odd, not even divisible by 2).
//
// == Optimal Configuration ==
// Block size: 512 threads (1D). Larger than reduction dim (511) — last thread idles.
//   Alternative: 256 threads with 2 elements/thread (511/256 = ~2).
//   512 is simpler: 1 element/thread, 1 thread masked.
// Grid: 1000 blocks (one per row). Non-power-of-2 grid is fine.
// Vector width: 1 (scalar fallback).
//   511 is odd, not divisible by 2, 4, or 8. Cannot safely vectorize without masking.
//   A smarter implementation could vec=2 for the first 510 elements and scalar for last 1,
//   but the simple approach is scalar.
//   CUB MemBoundScaling: items_per_thread = 1 * 4 / sizeof(f16) = 2, but we only
//     have 1 element/thread, so scaling doesn't help here.
// Elements per thread: 511 / 512 = ~1 element/thread. Thread 511 has no work.
//   With 256 threads: 511 / 256 = ~2 elements/thread.
// Reduction strategy: Warp shuffle reduction.
//   512 threads = 16 warps. Thread k < 511 loads input[row, k]. Thread 511 uses identity (0).
//   Warp shuffle reduces each warp, then smem reduces 16 partials.
// Coalescing analysis:
//   Input: row-major [1000, 511]. Row stride = 511 * sizeof(f16) = 1022 bytes.
//     1022 is NOT aligned to 128 bytes. Rows start at misaligned addresses.
//     Within a row: threads read consecutive f16 elements -> coalesced within a cache line,
//     but the row base may not be 128-byte aligned, causing split transactions.
//   Output: 1000 elements, trivially coalesced (one write per block).
// Masking: thread k >= 511 must use identity element (0.0) for the reduction.

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @row_sum_unaligned() {
  %cst = arith.constant 0.000000e+00 : f16
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1000x511xf16>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1000xf16>>
  %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [1000, 511], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1000x511xf16>> -> tensor<1000x511xf16>
  %3 = tensor.empty() : tensor<1000xf16>
  %4 = linalg.fill ins(%cst : f16) outs(%3 : tensor<1000xf16>) -> tensor<1000xf16>
  %5 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]}
    ins(%2 : tensor<1000x511xf16>) outs(%4 : tensor<1000xf16>) {
  ^bb0(%in: f16, %out: f16):
    %6 = arith.addf %in, %out : f16
    linalg.yield %6 : f16
  } -> tensor<1000xf16>
  iree_tensor_ext.dispatch.tensor.store %5, %1, offsets = [0], sizes = [1000], strides = [1] : tensor<1000xf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1000xf16>>
  return
}
