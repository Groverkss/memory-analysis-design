// Innermost dimension reduction (layer norm style)
// tensor<1024 x 4096 x f16> -> tensor<1024 x f16>
// d0=parallel, d1=reduction
// Expected: d0,d1 are in the same group per tensor dim.
//   Output dim 0 (size=1024): parallelizable
//   Input dim 1 (size=4096): NOT parallelizable (reduction in the generic)

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @innermost_reduction() {
  %cst = arith.constant 0.000000e+00 : f16
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024x4096xf16>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1024xf16>>
  %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [1024, 4096], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024x4096xf16>> -> tensor<1024x4096xf16>
  %3 = tensor.empty() : tensor<1024xf16>
  %4 = linalg.fill ins(%cst : f16) outs(%3 : tensor<1024xf16>) -> tensor<1024xf16>
  %5 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]}
    ins(%2 : tensor<1024x4096xf16>) outs(%4 : tensor<1024xf16>) {
  ^bb0(%in: f16, %out: f16):
    %6 = arith.addf %in, %out : f16
    linalg.yield %6 : f16
  } -> tensor<1024xf16>
  iree_tensor_ext.dispatch.tensor.store %5, %1, offsets = [0], sizes = [1024], strides = [1] : tensor<1024xf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1024xf16>>
  return
}
