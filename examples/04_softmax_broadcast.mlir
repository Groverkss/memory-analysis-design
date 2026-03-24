// Softmax: max reduction -> exp+broadcast -> sum reduction -> div+broadcast
// tensor<512x10240xf32> -> tensor<512x10240xf32>
// d0=parallel across all ops, d1=reduction in reductions, parallel in broadcasts
// Expected: d0 parallelizable, d1 NOT parallelizable

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @softmax_broadcast() {
  %c0 = arith.constant 0 : index
  %cst_ninf = arith.constant 0xFF800000 : f32
  %cst_zero = arith.constant 0.000000e+00 : f32
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<512x10240xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<512x10240xf32>>
  %in = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [512, 10240], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<512x10240xf32>> -> tensor<512x10240xf32>
  %empty_row = tensor.empty() : tensor<512xf32>
  %empty_full = tensor.empty() : tensor<512x10240xf32>

  // Op 1: max reduction along d1
  %fill_ninf = linalg.fill ins(%cst_ninf : f32) outs(%empty_row : tensor<512xf32>) -> tensor<512xf32>
  %max = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]}
    ins(%in : tensor<512x10240xf32>) outs(%fill_ninf : tensor<512xf32>) {
  ^bb0(%a: f32, %b: f32):
    %m = arith.maximumf %a, %b : f32
    linalg.yield %m : f32
  } -> tensor<512xf32>

  // Op 2: exp(x - max) with broadcast of max
  %exp = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%in, %max : tensor<512x10240xf32>, tensor<512xf32>) outs(%empty_full : tensor<512x10240xf32>) {
  ^bb0(%x: f32, %m: f32, %out: f32):
    %sub = arith.subf %x, %m : f32
    %e = math.exp %sub : f32
    linalg.yield %e : f32
  } -> tensor<512x10240xf32>

  // Op 3: sum reduction of exp along d1
  %fill_zero = linalg.fill ins(%cst_zero : f32) outs(%empty_row : tensor<512xf32>) -> tensor<512xf32>
  %sum = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]}
    ins(%exp : tensor<512x10240xf32>) outs(%fill_zero : tensor<512xf32>) {
  ^bb0(%a: f32, %b: f32):
    %s = arith.addf %a, %b : f32
    linalg.yield %s : f32
  } -> tensor<512xf32>

  // Op 4: div by sum with broadcast
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%exp, %sum : tensor<512x10240xf32>, tensor<512xf32>) outs(%empty_full : tensor<512x10240xf32>) {
  ^bb0(%e: f32, %s: f32, %out: f32):
    %d = arith.divf %e, %s : f32
    linalg.yield %d : f32
  } -> tensor<512x10240xf32>

  iree_tensor_ext.dispatch.tensor.store %result, %1, offsets = [0, 0], sizes = [512, 10240], strides = [1, 1] : tensor<512x10240xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<512x10240xf32>>
  return
}
