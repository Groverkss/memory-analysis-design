// Outer max reduction: tensor<2048x512xf32> -> tensor<512xf32>
// Find max along d0 (batch/outermost). Output is per-column max.
// This is the "column-wise max" used in some softmax variants and
// feature-scaling operations.
//
// Source pattern: PyTorch Reduce.cuh outer reduction with maximumf,
// CUTLASS TensorReductionAffineStrided (reduce along non-contiguous dim).
//
// == Optimal Configuration ==
// Block size: 256 threads (2D: block.x=32 parallel, block.y=8 reduction)
// Grid: ceil(512/32) = 16 workgroups
// Vector width: 4 (f32, 128-bit loads along d1, the contiguous dim)
// Elements per thread: 2048/8 = 256 d0 iterations
//
// Thread mapping:
//   threadIdx.x (32): d1 (parallel, contiguous). Coalesced reads.
//   threadIdx.y (8): d0 (reduction, strided).
//   Each thread reads d0 = threadIdx.y, threadIdx.y+8, ..., 2048.
//
// Memory access: Coalesced along d1 within each d0 step.
// Stride along d0 = 512 elements = 2048 bytes.
//
// Reduction: block.y tree reduction in shared memory using maximumf.
// Shared memory: 32 * 8 * sizeof(f32) = 1024 bytes.
// Init: -inf for max reduction.

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @outer_max_reduction() {
  %cst = arith.constant 0xFF800000 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048x512xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<512xf32>>
  %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [2048, 512], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048x512xf32>> -> tensor<2048x512xf32>
  %3 = tensor.empty() : tensor<512xf32>
  %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<512xf32>) -> tensor<512xf32>
  %5 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d1)>],
    iterator_types = ["reduction", "parallel"]}
    ins(%2 : tensor<2048x512xf32>) outs(%4 : tensor<512xf32>) {
  ^bb0(%in: f32, %out: f32):
    %6 = arith.maximumf %in, %out : f32
    linalg.yield %6 : f32
  } -> tensor<512xf32>
  iree_tensor_ext.dispatch.tensor.store %5, %1, offsets = [0], sizes = [512], strides = [1] : tensor<512xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<512xf32>>
  return
}
