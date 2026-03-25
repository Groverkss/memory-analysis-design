// Outermost reduction with odd parallel dim size (not power of 2)
// tensor<4096 x 511 x f16> -> tensor<511 x f16>
// P0=511 < 512 (coalescing target). Can't fully satisfy coalescing.

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @outermost_reduction_odd() {
  %cst = arith.constant 0.000000e+00 : f16
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x511xf16>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<511xf16>>
  %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [4096, 511], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x511xf16>> -> tensor<4096x511xf16>
  %3 = tensor.empty() : tensor<511xf16>
  %4 = linalg.fill ins(%cst : f16) outs(%3 : tensor<511xf16>) -> tensor<511xf16>
  %5 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d1)>],
    iterator_types = ["reduction", "parallel"]}
    ins(%2 : tensor<4096x511xf16>) outs(%4 : tensor<511xf16>) {
  ^bb0(%in: f16, %out: f16):
    %6 = arith.addf %in, %out : f16
    linalg.yield %6 : f16
  } -> tensor<511xf16>
  iree_tensor_ext.dispatch.tensor.store %5, %1, offsets = [0], sizes = [511], strides = [1] : tensor<511xf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<511xf16>>
  return
}
