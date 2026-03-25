// Outer (column-wise) sum reduction: tensor<4096x8192xf16> -> tensor<8192xf16>
// Reduce along d0 (outermost, strided). d1 (8192) is parallel and contiguous.
//
// Source pattern: PyTorch Reduce.cuh outer reduction, BatchNorm channel-wise stats,
// MIOpen BatchNorm training (reduce over N*H*W, keep C).
//
// == Optimal Configuration ==
// Block size: 512 threads (2D: block.x=32 spatial, block.y=16 reduction)
// Vector width: 8 (f16, 128-bit loads)
// Output vectorization: vec_size=8 (consecutive threads write consecutive d1 elements)
// Elements per thread: Each thread reads 4096/16 = 256 elements along d0 (reduction)
// Grid: ceil(8192/32) = 256 workgroups (tiling along d1)
//
// Thread mapping:
//   threadIdx.x (32): maps to d1 (output/parallel dim). Consecutive threads read
//     consecutive d1 positions -> coalesced reads along contiguous dim.
//   threadIdx.y (16): maps to d0 (reduction dim). Each group of 16 threads cooperates
//     on the d0 reduction for the same d1 position.
//
// Memory access: Input reads are coalesced (threads in x access consecutive d1).
// Each thread strides along d0 with step block.y (16), reading 4096/16 = 256 iterations.
// This is a STRIDED pattern along d0 but coalesced along d1 within each row.
//
// Reduction: block.y threads cooperate via shared memory tree reduction (4 steps).
// Shared memory: block.x * block.y * sizeof(f32) = 32*16*4 = 2048 bytes for accumulators.

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @outer_reduction_f16() {
  %cst = arith.constant 0.000000e+00 : f16
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x8192xf16>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8192xf16>>
  %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [4096, 8192], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x8192xf16>> -> tensor<4096x8192xf16>
  %3 = tensor.empty() : tensor<8192xf16>
  %4 = linalg.fill ins(%cst : f16) outs(%3 : tensor<8192xf16>) -> tensor<8192xf16>
  %5 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d1)>],
    iterator_types = ["reduction", "parallel"]}
    ins(%2 : tensor<4096x8192xf16>) outs(%4 : tensor<8192xf16>) {
  ^bb0(%in: f16, %out: f16):
    %6 = arith.addf %in, %out : f16
    linalg.yield %6 : f16
  } -> tensor<8192xf16>
  iree_tensor_ext.dispatch.tensor.store %5, %1, offsets = [0], sizes = [8192], strides = [1] : tensor<8192xf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8192xf16>>
  return
}
