// 3D outer reduction: tensor<256x32x4096xf16> -> tensor<32x4096xf16>
// Reduce along d0 (batch dimension). d1 and d2 are parallel.
// This is the BatchNorm pattern: reduce over batch, keep channel and spatial.
//
// Source pattern: Apex welford.cu NHWC (reduce over N, keep C*H*W),
// MIOpen BatchNorm spatial multiple variant.
//
// == Optimal Configuration ==
// Block size: 256 threads (2D: block.x=32, block.y=8)
// Grid: ceil(32*4096 / 32) = 4096 workgroups (tiling d1*d2 fused)
// Vector width: 8 (f16, 128-bit loads along contiguous d2)
// Elements per thread: 256/8 = 32 d0 iterations per thread
//
// Thread mapping:
//   Fuse d1 and d2 into one parallel dimension of size 32*4096 = 131072.
//   threadIdx.x (32): maps to the fused parallel dim -> coalesced along d2.
//   threadIdx.y (8): cooperates on d0 reduction.
//   Each (block.x thread, block.y thread) pair handles one fused_parallel_pos
//   and iterates over d0 with stride block.y.
//
// Memory access: Coalesced along d2 (contiguous). Strided along d0 by d1*d2 elements.
// Reduction: block.y tree reduction in shared memory.
// Shared memory: 32 * 8 * sizeof(f32) = 1024 bytes.

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @outer_reduction_3d() {
  %cst = arith.constant 0.000000e+00 : f16
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x32x4096xf16>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32x4096xf16>>
  %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [256, 32, 4096], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x32x4096xf16>> -> tensor<256x32x4096xf16>
  %3 = tensor.empty() : tensor<32x4096xf16>
  %4 = linalg.fill ins(%cst : f16) outs(%3 : tensor<32x4096xf16>) -> tensor<32x4096xf16>
  %5 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                     affine_map<(d0, d1, d2) -> (d1, d2)>],
    iterator_types = ["reduction", "parallel", "parallel"]}
    ins(%2 : tensor<256x32x4096xf16>) outs(%4 : tensor<32x4096xf16>) {
  ^bb0(%in: f16, %out: f16):
    %6 = arith.addf %in, %out : f16
    linalg.yield %6 : f16
  } -> tensor<32x4096xf16>
  iree_tensor_ext.dispatch.tensor.store %5, %1, offsets = [0, 0], sizes = [32, 4096], strides = [1, 1] : tensor<32x4096xf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32x4096xf16>>
  return
}
