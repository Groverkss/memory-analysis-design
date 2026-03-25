// global_reduction.mlir
// Sum all elements: tensor<16777216xf32> -> tensor<f32>
// Pattern: CUB multi-block reduce. Full-tensor reduction to a scalar.
// In practice this requires two-pass: partial sums per block, then reduce partials.
// At the MLIR level we express the single logical reduction; the compiler must
// decompose into multi-block if needed.
//
// == Optimal Configuration ==
// Block size: 256 threads.
// Grid: occupancy * SM_count * 5 blocks. Example: 4 active blocks/SM * 108 SMs * 5
//   = 2160 blocks (A100). The factor of 5 is CUB's heuristic to saturate memory BW.
// Vector width: 4 (f32 vec4 = 16 bytes).
// Elements per thread: 16777216 / (2160 * 256) = ~30 elements/thread.
//   With vec=4: ~8 vectorized iterations per thread.
//   CUB MemBoundScaling: items_per_thread = 30 * 4 / sizeof(f32) = 30 (no scaling for f32).
// Reduction strategy: Multi-block (two-pass).
//   Pass 1: Each of ~2160 blocks reduces its chunk via warp shuffle -> one partial sum/block.
//     Within a block: each thread accumulates ~30 elements in a loop.
//     Then warp shuffle reduction, then 8 warp partials via smem.
//     Block writes 1 partial sum to global memory.
//   Pass 2: A single block of 256 threads reduces ~2160 partial sums -> final scalar.
//     2160 / 256 = ~9 elements/thread, then warp shuffle + smem.
// Coalescing: 1D tensor, consecutive threads read consecutive elements -> perfect coalescing.
// Note: Single-block would leave 99.5% of the GPU idle. Multi-block is essential.

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @global_reduction() {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16777216xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<f32>>
  %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0], sizes = [16777216], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16777216xf32>> -> tensor<16777216xf32>
  %3 = tensor.empty() : tensor<f32>
  %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<f32>) -> tensor<f32>
  %5 = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>,
                     affine_map<(d0) -> ()>],
    iterator_types = ["reduction"]}
    ins(%2 : tensor<16777216xf32>) outs(%4 : tensor<f32>) {
  ^bb0(%in: f32, %out: f32):
    %6 = arith.addf %in, %out : f32
    linalg.yield %6 : f32
  } -> tensor<f32>
  iree_tensor_ext.dispatch.tensor.store %5, %1, offsets = [], sizes = [], strides = [] : tensor<f32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<f32>>
  return
}
