// outer_reduction.mlir
// Sum along d0 (outer dim): tensor<4096x1024xf32> -> tensor<1024xf32>
// Pattern: PyTorch Reduce.cuh outer reduction. block.x maps to spatial (output),
// block.y cooperates on the reduction dimension.
//
// == Optimal Configuration ==
// Block size: 2D block (32, 16) = 512 threads.
//   threadIdx.x = 32: maps to output dimension d1. Warp-aligned for coalescing.
//   threadIdx.y = 16: cooperates on reduction dimension d0.
// Grid: (1024 / 32, 1) = (32, 1) blocks. Each block produces 32 output elements.
// Vector width: 4 (threadIdx.x covers 32 elements, but can vectorize output writes;
//   for the input, consecutive threadIdx.x threads read consecutive d1 elements
//   within the same row -> coalesced. vec=4 on the output store).
// Elements per thread (reduction): 4096 / 16 = 256 rows per threadIdx.y thread.
//   Each thread loops over 256 rows, accumulating partial sums for its d1 position.
// Reduction strategy: Shared memory tree reduction along threadIdx.y.
//   Each threadIdx.y lane accumulates a partial sum over its 256 rows.
//   Then 16 values are reduced via smem with log2(16) = 4 barriers.
//   threadIdx.y == 0 writes the final result.
// Coalescing analysis:
//   Input: row-major [4096, 1024]. Thread (tx, ty) reads input[ty*stride + row, tx].
//     Within a warp, threads have consecutive tx -> reading consecutive columns
//     in the same row -> perfectly coalesced.
//   Output: threadIdx.x maps to consecutive output elements -> coalesced writes.
// Thread mapping:
//   threadIdx.x -> d1 (output spatial dimension, stride-1 in memory)
//   threadIdx.y -> d0 (reduction dimension, each ty handles a chunk of rows)

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @outer_reduction() {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x1024xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1024xf32>>
  %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [4096, 1024], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x1024xf32>> -> tensor<4096x1024xf32>
  %3 = tensor.empty() : tensor<1024xf32>
  %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<1024xf32>) -> tensor<1024xf32>
  %5 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d1)>],
    iterator_types = ["reduction", "parallel"]}
    ins(%2 : tensor<4096x1024xf32>) outs(%4 : tensor<1024xf32>) {
  ^bb0(%in: f32, %out: f32):
    %6 = arith.addf %in, %out : f32
    linalg.yield %6 : f32
  } -> tensor<1024xf32>
  iree_tensor_ext.dispatch.tensor.store %5, %1, offsets = [0], sizes = [1024], strides = [1] : tensor<1024xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1024xf32>>
  return
}
