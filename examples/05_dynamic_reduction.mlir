// Dynamic innermost reduction (e.g., layer norm with dynamic sequence length)
// tensor<?x?xf16> -> tensor<?xf16>
// d0=parallel, d1=reduction
// Expected: P0 = ? (parallel), R0 = ? (reduction)

#pipeline_layout = #hal.pipeline.layout<constants = 2, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @dynamic_reduction() {
  %cst = arith.constant 0.000000e+00 : f16
  %c0 = arith.constant 0 : index
  %dim0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %dim1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf16>>{%dim0, %dim1}
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xf16>>{%dim0}
  %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [%dim0, %dim1], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf16>>{%dim0, %dim1} -> tensor<?x?xf16>
  %3 = tensor.empty(%dim0) : tensor<?xf16>
  %4 = linalg.fill ins(%cst : f16) outs(%3 : tensor<?xf16>) -> tensor<?xf16>
  %5 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]}
    ins(%2 : tensor<?x?xf16>) outs(%4 : tensor<?xf16>) {
  ^bb0(%in: f16, %out: f16):
    %6 = arith.addf %in, %out : f16
    linalg.yield %6 : f16
  } -> tensor<?xf16>
  iree_tensor_ext.dispatch.tensor.store %5, %1, offsets = [0], sizes = [%dim0], strides = [1] : tensor<?xf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xf16>>{%dim0}
  return
}
