// Pure elementwise: f16 -> f32 with scaling
// tensor<1024x4096xf16> -> tensor<1024x4096xf32>
// All dims parallel. Coalescing on both input and output innermost dim.

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @elementwise() {
  %cst = arith.constant 2.0 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024x4096xf16>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1024x4096xf32>>
  %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [1024, 4096], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024x4096xf16>> -> tensor<1024x4096xf16>
  %3 = tensor.empty() : tensor<1024x4096xf32>
  %4 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%2 : tensor<1024x4096xf16>) outs(%3 : tensor<1024x4096xf32>) {
  ^bb0(%in: f16, %out: f32):
    %5 = arith.extf %in : f16 to f32
    %6 = arith.mulf %5, %cst : f32
    linalg.yield %6 : f32
  } -> tensor<1024x4096xf32>
  iree_tensor_ext.dispatch.tensor.store %4, %1, offsets = [0, 0], sizes = [1024, 4096], strides = [1, 1] : tensor<1024x4096xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1024x4096xf32>>
  return
}
