// inner_reduction_f16_large.mlir
// Sum reduction along inner dim: tensor<1024x65536xf16> -> tensor<1024xf16>
// Pattern: Very large inner reduction. Needs looping within each thread.
//
// == Optimal Configuration ==
// Block size: 512 threads (1D).
// Grid: 1024 blocks (one per row).
// Vector width: 8 (f16 vec8 = 16 bytes, fills 128-bit load).
// Elements per thread: 65536 / 512 = 128 raw elements per thread.
//   With vec=8: 128 / 8 = 16 vectorized loop iterations per thread.
//   CUB MemBoundScaling: items_per_thread = nominal * 4 / sizeof(f16) = nominal * 2.
//     Base nominal ~16, scaled = 16 * 2 = 32. We use 128 (capped by problem size).
// Reduction strategy: Warp shuffle reduction.
//   Each thread loops over its 16 vec8 chunks, accumulating a local partial sum.
//   Then warp shuffle (__shfl_down) reduces 32 threads to 1 warp partial.
//   512 threads = 16 warps -> 16 warp partials reduced via smem or final warp.
// Coalescing: thread k reads elements [row, k*vec], [row, k*vec + 512*vec], etc.
//   Stride-1 access in the inner dim -> perfectly coalesced.
//   A warp of 32 threads loads 32 * 8 = 256 contiguous f16 values per iteration.
// Note: 512 threads chosen over 256 to reduce loop iterations (16 vs 32) and
//   increase memory-level parallelism. 1024 threads would also work but may
//   hurt occupancy on some GPUs.

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @inner_reduction_f16_large() {
  %cst = arith.constant 0.000000e+00 : f16
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024x65536xf16>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1024xf16>>
  %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [1024, 65536], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024x65536xf16>> -> tensor<1024x65536xf16>
  %3 = tensor.empty() : tensor<1024xf16>
  %4 = linalg.fill ins(%cst : f16) outs(%3 : tensor<1024xf16>) -> tensor<1024xf16>
  %5 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]}
    ins(%2 : tensor<1024x65536xf16>) outs(%4 : tensor<1024xf16>) {
  ^bb0(%in: f16, %out: f16):
    %6 = arith.addf %in, %out : f16
    linalg.yield %6 : f16
  } -> tensor<1024xf16>
  iree_tensor_ext.dispatch.tensor.store %5, %1, offsets = [0], sizes = [1024], strides = [1] : tensor<1024xf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1024xf16>>
  return
}
