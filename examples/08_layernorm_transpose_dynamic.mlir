// Fused layer norm with transposed output, dynamic shapes.
// tensor<?x?x?x?xf16> -> tensor<?x?x?x?xf32> (transposed)
// Same structure as 07 but all dims dynamic.

#pipeline_layout = #hal.pipeline.layout<constants = 4, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @layernorm_transpose_dynamic() {
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant 1.638400e+05 : f32
  %cst_1 = arith.constant 9.99999974E-6 : f32
  %c0 = arith.constant 0 : index
  %d0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %d1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
  %d2 = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : index
  %d3 = hal.interface.constant.load layout(#pipeline_layout) ordinal(3) : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x?x?xf16>>{%d0, %d1, %d2, %d3}
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?x?x?xf32>>{%d2, %d3, %d0, %d1}
  %in = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [%d0, %d1, %d2, %d3], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x?x?xf16>>{%d0, %d1, %d2, %d3} -> tensor<?x?x?x?xf16>
  %empty_full = tensor.empty(%d0, %d1, %d2, %d3) : tensor<?x?x?x?xf32>
  %empty_small = tensor.empty(%d0, %d1) : tensor<?x?xf32>
  %empty_transposed = tensor.empty(%d2, %d3, %d0, %d1) : tensor<?x?x?x?xf32>

  // Op 1: extf
  %ext = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    ins(%in : tensor<?x?x?x?xf16>) outs(%empty_full : tensor<?x?x?x?xf32>) {
  ^bb0(%a: f16, %b: f32):
    %v = arith.extf %a : f16 to f32
    linalg.yield %v : f32
  } -> tensor<?x?x?x?xf32>

  // Op 2: fill
  %fill = linalg.fill ins(%cst : f32) outs(%empty_small : tensor<?x?xf32>) -> tensor<?x?xf32>

  // Op 3: mean reduction
  %mean = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel", "reduction", "reduction"]}
    ins(%ext : tensor<?x?x?x?xf32>) outs(%fill : tensor<?x?xf32>) {
  ^bb0(%a: f32, %b: f32):
    %v = arith.addf %a, %b : f32
    linalg.yield %v : f32
  } -> tensor<?x?xf32>

  // Op 4: divf
  %div = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%mean : tensor<?x?xf32>) outs(%empty_small : tensor<?x?xf32>) {
  ^bb0(%a: f32, %b: f32):
    %v = arith.divf %a, %cst_0 : f32
    linalg.yield %v : f32
  } -> tensor<?x?xf32>

  // Op 5: variance reduction
  %var = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel", "reduction", "reduction"]}
    ins(%ext, %div : tensor<?x?x?x?xf32>, tensor<?x?xf32>) outs(%fill : tensor<?x?xf32>) {
  ^bb0(%a: f32, %m: f32, %b: f32):
    %sub = arith.subf %a, %m : f32
    %sq = arith.mulf %sub, %sub : f32
    %add = arith.addf %sq, %b : f32
    linalg.yield %add : f32
  } -> tensor<?x?xf32>

  // Op 6: normalize + transpose
  %norm = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1)>,
                     affine_map<(d0, d1, d2, d3) -> (d2, d3, d0, d1)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    ins(%in, %div, %var : tensor<?x?x?x?xf16>, tensor<?x?xf32>, tensor<?x?xf32>) outs(%empty_transposed : tensor<?x?x?x?xf32>) {
  ^bb0(%x: f16, %m: f32, %v: f32, %out: f32):
    %dv = arith.divf %v, %cst_0 : f32
    %eps = arith.addf %dv, %cst_1 : f32
    %rs = math.rsqrt %eps : f32
    %xf = arith.extf %x : f16 to f32
    %sub = arith.subf %xf, %m : f32
    %mul = arith.mulf %sub, %rs : f32
    linalg.yield %mul : f32
  } -> tensor<?x?x?x?xf32>

  iree_tensor_ext.dispatch.tensor.store %norm, %1, offsets = [0, 0, 0, 0], sizes = [%d2, %d3, %d0, %d1], strides = [1, 1, 1, 1] : tensor<?x?x?x?xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?x?x?xf32>>{%d2, %d3, %d0, %d1}
  return
}
