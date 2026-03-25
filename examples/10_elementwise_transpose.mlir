// Elementwise with transposed store: f16 -> f32, output is transposed
// tensor<1024x4096xf16> -> tensor<4096x1024xf32>
// Input contiguous dim = d1 (4096), output contiguous dim = d0 (1024).
// Coalescing conflict between input and output.

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @elementwise_transpose() {
  %cst = arith.constant 2.0 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024x4096xf16>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4096x1024xf32>>
  %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [1024, 4096], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024x4096xf16>> -> tensor<1024x4096xf16>
  %3 = tensor.empty() : tensor<4096x1024xf32>
  %4 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d1, d0)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%2 : tensor<1024x4096xf16>) outs(%3 : tensor<4096x1024xf32>) {
  ^bb0(%in: f16, %out: f32):
    %5 = arith.extf %in : f16 to f32
    %6 = arith.mulf %5, %cst : f32
    linalg.yield %6 : f32
  } -> tensor<4096x1024xf32>
  iree_tensor_ext.dispatch.tensor.store %4, %1, offsets = [0, 0], sizes = [4096, 1024], strides = [1, 1] : tensor<4096x1024xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4096x1024xf32>>
  return
}
