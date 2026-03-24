// Fused layer norm: extf -> mean reduction -> divf -> variance reduction -> normalize
// tensor<2x32x10x16384xf16> -> tensor<2x32x10x16384xf32>
// d0,d1 are parallel across ALL ops; d2,d3 are reduction in some, parallel in others.
// Expected dimension groups:
//   - Groups containing d0-related dims: parallelizable (size=2)
//   - Groups containing d1-related dims: parallelizable (size=32)
//   - Groups containing d2-related dims: NOT parallelizable (reduction in mean/var)
//   - Groups containing d3-related dims: NOT parallelizable (reduction in mean/var)

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @layernorm_fused() {
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant 1.638400e+05 : f32
  %cst_1 = arith.constant 9.99999974E-6 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x32x10x16384xf16>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x32x10x16384xf16>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x32x10x16384xf32>>
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [2, 32, 10, 16384], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x32x10x16384xf16>> -> tensor<2x32x10x16384xf16>
  %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [2, 32, 10, 16384], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x32x10x16384xf16>> -> tensor<2x32x10x16384xf16>
  %5 = tensor.empty() : tensor<2x32x10x16384xf32>
  %6 = tensor.empty() : tensor<2x32xf32>

  // Op 1: extf (elementwise f16 -> f32)
  %7 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    ins(%3 : tensor<2x32x10x16384xf16>) outs(%5 : tensor<2x32x10x16384xf32>) {
  ^bb0(%in: f16, %out: f32):
    %13 = arith.extf %in : f16 to f32
    linalg.yield %13 : f32
  } -> tensor<2x32x10x16384xf32>

  // Op 2: fill for accumulator
  %8 = linalg.fill ins(%cst : f32) outs(%6 : tensor<2x32xf32>) -> tensor<2x32xf32>

  // Op 3: mean reduction (reduce d2, d3)
  %9 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel", "reduction", "reduction"]}
    ins(%7 : tensor<2x32x10x16384xf32>) outs(%8 : tensor<2x32xf32>) {
  ^bb0(%in: f32, %out: f32):
    %13 = arith.addf %in, %out : f32
    linalg.yield %13 : f32
  } -> tensor<2x32xf32>

  // Op 4: divf to get mean
  %10 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%9 : tensor<2x32xf32>) outs(%6 : tensor<2x32xf32>) {
  ^bb0(%in: f32, %out: f32):
    %13 = arith.divf %in, %cst_0 : f32
    linalg.yield %13 : f32
  } -> tensor<2x32xf32>

  // Op 5: variance reduction (reduce d2, d3, with mean broadcast)
  %11 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel", "reduction", "reduction"]}
    ins(%7, %10 : tensor<2x32x10x16384xf32>, tensor<2x32xf32>) outs(%8 : tensor<2x32xf32>) {
  ^bb0(%in: f32, %in_2: f32, %out: f32):
    %13 = arith.subf %in, %in_2 : f32
    %14 = arith.mulf %13, %13 : f32
    %15 = arith.addf %14, %out : f32
    linalg.yield %15 : f32
  } -> tensor<2x32xf32>

  // Op 6: normalization (elementwise with broadcasts)
  %12 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    ins(%4, %10, %11 : tensor<2x32x10x16384xf16>, tensor<2x32xf32>, tensor<2x32xf32>) outs(%5 : tensor<2x32x10x16384xf32>) {
  ^bb0(%in: f16, %in_2: f32, %in_3: f32, %out: f32):
    %13 = arith.divf %in_3, %cst_0 : f32
    %14 = arith.addf %13, %cst_1 : f32
    %15 = math.rsqrt %14 : f32
    %16 = arith.extf %in : f16 to f32
    %17 = arith.subf %16, %in_2 : f32
    %18 = arith.mulf %17, %15 : f32
    linalg.yield %18 : f32
  } -> tensor<2x32x10x16384xf32>

  iree_tensor_ext.dispatch.tensor.store %12, %2, offsets = [0, 0, 0, 0], sizes = [2, 32, 10, 16384], strides = [1, 1, 1, 1] : tensor<2x32x10x16384xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x32x10x16384xf32>>
  return
}
