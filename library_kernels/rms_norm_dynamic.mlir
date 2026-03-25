// RMSNorm with fully dynamic batch and hidden dimensions
// Pattern: Dynamic shapes. Config must handle unknown sizes at compile time.
// tensor<?x?xf16> -> tensor<?x?xf16>, weights tensor<?xf16>
// Uses IntegerRangeAnalysis upper bounds for heuristic decisions.
//
// == Optimal Configuration ==
// hidden_dim=?, batch=?, f16 input/output, f32 accumulation
// Without static sizes, config must use resolved upper bounds from
// IntegerRangeAnalysis. Typical strategy:
//   - Assume worst-case threads_per_row = 256 (covers up to 256*8=2048 elems)
//   - rows_per_block = 1 (conservative, since hidden could be large)
//   - workgroup_size = 256
//   - vector_width = 8 (f16)
//   - Masking is always needed (hidden_dim may not be multiple of 8*256)
//   - num_workgroups = ceil(batch / rows_per_block) = batch (dynamic)
// reduction_strategy: 256 threads -> shuffle + shared memory (8 warps)
// coalescing: d1 innermost assumed contiguous (row-major layout).
// The key challenge is that elements_per_thread is unknown, so the
// reduction loop bound is dynamic. No compile-time register allocation
// optimization is possible.

#pipeline_layout = #hal.pipeline.layout<constants = 2, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @rms_norm_dynamic() {
  %cst = arith.constant 0.000000e+00 : f32
  %cst_eps = arith.constant 9.99999974E-6 : f32
  %c0 = arith.constant 0 : index
  %dim0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %dim1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf16>>{%dim0, %dim1}
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?xf16>>{%dim1}
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?xf16>>{%dim0, %dim1}
  %input = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [%dim0, %dim1], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf16>>{%dim0, %dim1} -> tensor<?x?xf16>
  %weight = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0], sizes = [%dim1], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?xf16>>{%dim1} -> tensor<?xf16>
  %empty_2d = tensor.empty(%dim0, %dim1) : tensor<?x?xf32>
  %empty_1d = tensor.empty(%dim0) : tensor<?xf32>
  %empty_out = tensor.empty(%dim0, %dim1) : tensor<?x?xf16>

  // Convert dim1 to f32 for division
  %dim1_idx = arith.index_cast %dim1 : index to i64
  %dim1_f32 = arith.uitofp %dim1_idx : i64 to f32

  // Op 1: extf + square
  %squared = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%input : tensor<?x?xf16>) outs(%empty_2d : tensor<?x?xf32>) {
  ^bb0(%in: f16, %out: f32):
    %v = arith.extf %in : f16 to f32
    %sq = arith.mulf %v, %v : f32
    linalg.yield %sq : f32
  } -> tensor<?x?xf32>

  // Op 2: fill
  %acc_init = linalg.fill ins(%cst : f32) outs(%empty_1d : tensor<?xf32>) -> tensor<?xf32>

  // Op 3: reduce sum of squares
  %sum_sq = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]}
    ins(%squared : tensor<?x?xf32>) outs(%acc_init : tensor<?xf32>) {
  ^bb0(%in: f32, %out: f32):
    %s = arith.addf %in, %out : f32
    linalg.yield %s : f32
  } -> tensor<?xf32>

  // Op 4: rsqrt(mean + eps)
  %rrms = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>,
                     affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]}
    ins(%sum_sq : tensor<?xf32>) outs(%empty_1d : tensor<?xf32>) {
  ^bb0(%in: f32, %out: f32):
    %mean = arith.divf %in, %dim1_f32 : f32
    %shifted = arith.addf %mean, %cst_eps : f32
    %r = math.rsqrt %shifted : f32
    linalg.yield %r : f32
  } -> tensor<?xf32>

  // Op 5: normalize + weight
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>,
                     affine_map<(d0, d1) -> (d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%input, %rrms, %weight : tensor<?x?xf16>, tensor<?xf32>, tensor<?xf16>) outs(%empty_out : tensor<?x?xf16>) {
  ^bb0(%in: f16, %scale: f32, %w: f16, %out: f16):
    %xf = arith.extf %in : f16 to f32
    %wf = arith.extf %w : f16 to f32
    %normed = arith.mulf %xf, %scale : f32
    %weighted = arith.mulf %normed, %wf : f32
    %r = arith.truncf %weighted : f32 to f16
    linalg.yield %r : f16
  } -> tensor<?x?xf16>

  iree_tensor_ext.dispatch.tensor.store %result, %2, offsets = [0, 0], sizes = [%dim0, %dim1], strides = [1, 1] : tensor<?x?xf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?xf16>>{%dim0, %dim1}
  return
}
