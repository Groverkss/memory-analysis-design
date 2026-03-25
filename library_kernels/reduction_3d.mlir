// reduction_3d.mlir
// Sum along d1 (middle dim): tensor<8x256x4096xf32> -> tensor<8x4096xf32>
// Pattern: Middle-dim reduction (like batch norm stats). Neither purely inner nor outer.
// d2 is contiguous in memory, d1 is the reduction dim, d0 is the batch dim.
//
// == Optimal Configuration ==
// Block size: 2D block (32, 8) = 256 threads.
//   threadIdx.x = 32: maps to d2 (innermost/contiguous dim, parallel). Coalesced.
//   threadIdx.y = 8: cooperates on d1 (middle dim, reduction).
// Grid: (4096 / 32, 8) = (128, 8) blocks. 128 blocks along d2, 8 blocks for batch d0.
//   Each block produces 32 output elements for one batch.
// Vector width: 4 (f32 vec4 = 16 bytes). threadIdx.x handles 32 elements, and
//   the 32 threads can be further vectorized along d2.
//   With vec=4: each threadIdx.x handles 32 positions, but blocks tile d2 by 32,
//   so effectively 32 * vec4 isn't needed. vec=4 applies if block tile > 32.
//   Simpler: vec=1 with 32 threads covering 32 d2 elements. vec=4 would mean
//   block covers 128 d2 elements with 32 threads, grid = (4096/128, 8) = (32, 8).
// Elements per thread (reduction): 256 / 8 = 32 iterations along d1 per threadIdx.y.
// Reduction strategy: Shared memory tree reduction along threadIdx.y.
//   Each threadIdx.y lane accumulates partial sums over its 32 slices of d1.
//   smem[ty][tx] stores partials. log2(8) = 3 barriers to reduce.
//   threadIdx.y == 0 writes the 32 output elements.
// Coalescing analysis:
//   Input: row-major [8, 256, 4096]. Innermost stride is 1 along d2.
//     Thread (tx, ty) reads input[batch, ty*32 + d1_iter, block_d2_offset + tx].
//     Consecutive tx threads read consecutive d2 elements -> perfectly coalesced.
//     Stride between d1 slices = 4096 * sizeof(f32) = 16KB. Large stride, but
//     each d1 step is a separate iteration, not adjacent in the access pattern.
//   Output: [8, 4096]. threadIdx.x maps to consecutive d2 -> coalesced writes.
// Note: d1 reduction is strided because d2 (size 4096) sits between d1 elements
//   in memory. This is fine for GPU because each d1 iteration is a separate load
//   and the L2 cache handles the reuse of x-dimension neighbors.

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @reduction_3d() {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x256x4096xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8x4096xf32>>
  %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [8, 256, 4096], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x256x4096xf32>> -> tensor<8x256x4096xf32>
  %3 = tensor.empty() : tensor<8x4096xf32>
  %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<8x4096xf32>) -> tensor<8x4096xf32>
  %5 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                     affine_map<(d0, d1, d2) -> (d0, d2)>],
    iterator_types = ["parallel", "reduction", "parallel"]}
    ins(%2 : tensor<8x256x4096xf32>) outs(%4 : tensor<8x4096xf32>) {
  ^bb0(%in: f32, %out: f32):
    %6 = arith.addf %in, %out : f32
    linalg.yield %6 : f32
  } -> tensor<8x4096xf32>
  iree_tensor_ext.dispatch.tensor.store %5, %1, offsets = [0, 0], sizes = [8, 4096], strides = [1, 1] : tensor<8x4096xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8x4096xf32>>
  return
}
