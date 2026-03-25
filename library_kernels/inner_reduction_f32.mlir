// inner_reduction_f32.mlir
// Sum reduction along inner dim: tensor<8192x256xf32> -> tensor<8192xf32>
// Pattern: CUB inner reduce. Small reduction dim (256), many rows (8192).
//
// == Optimal Configuration ==
// Block size: 256 threads (1D). Matches reduction dim exactly.
// Grid: 8192 blocks (one per row).
// Vector width: 4 (f32 vec4 = 16 bytes, fills 128-bit load).
// Elements per thread: 256 / 256 = 1 element (after vectorization: 4 raw elements
//   loaded as vec4, but logically 256/256=1 "iteration" per thread).
//   CUB MemBoundScaling: items_per_thread = 1 * 4 / sizeof(f32) = 1.
// Reduction strategy: Warp shuffle reduction. 256 threads = 8 warps.
//   Each thread loads its element(s), partial sum via __shfl_down within warp,
//   then 8 warp-partial results reduced via shared memory or final warp shuffle.
// Coalescing: thread k reads element [row, k] — contiguous in memory (row-major).
//   Perfect coalescing: 32 threads in a warp read 32 consecutive f32 values.
// Note: Reduction dim 256 is small enough for single-pass (no looping needed).

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @inner_reduction_f32() {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8192x256xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8192xf32>>
  %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [8192, 256], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8192x256xf32>> -> tensor<8192x256xf32>
  %3 = tensor.empty() : tensor<8192xf32>
  %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<8192xf32>) -> tensor<8192xf32>
  %5 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]}
    ins(%2 : tensor<8192x256xf32>) outs(%4 : tensor<8192xf32>) {
  ^bb0(%in: f32, %out: f32):
    %6 = arith.addf %in, %out : f32
    linalg.yield %6 : f32
  } -> tensor<8192xf32>
  iree_tensor_ext.dispatch.tensor.store %5, %1, offsets = [0], sizes = [8192], strides = [1] : tensor<8192xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8192xf32>>
  return
}
