// RMSNorm with small hidden dimension (128), large batch (8192)
// Pattern: Quack/FlashInfer small hidden dim -> multiple rows per block
// tensor<8192x128xf16> -> tensor<8192x128xf16>, weights tensor<128xf16>
// Computation: x * rsqrt(mean(x^2) + eps) * weight
//
// == Optimal Configuration ==
// hidden_dim=128, batch=8192, f16 input/output, f32 accumulation
// threads_per_row = 16 (128 / 8 vec_elems = 16 threads cover one row)
// vector_width = 8 (128 bits / 16 bits = 8 f16 elements)
// elements_per_thread = 8 (128 / 16 = 8, exactly one vector load)
// rows_per_block = 16 (256 threads / 16 threads_per_row = 16 rows)
// workgroup_size = 256 (16 threads_per_row * 16 rows_per_block)
// num_workgroups = 512 (8192 / 16)
// reduction_strategy: single warp-row, 16 threads need only 4 shuffle rounds
//   (log2(16) = 4), no shared memory needed since threads_per_row <= 32
// coalescing: d1 (hidden) is innermost and contiguous. 16 threads * 8 elems
//   * 2 bytes = 256 bytes per row, 16 rows = 4096 bytes per WG load.
//   Well above 128-byte coalescing target.
// register pressure: each thread holds 8 f16 inputs + 8 f32 intermediates
//   = 16 + 32 = 48 bytes, very light.

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @rms_norm_small_hidden() {
  %cst = arith.constant 0.000000e+00 : f32
  %cst_eps = arith.constant 9.99999974E-6 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8192x128xf16>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128xf16>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8192x128xf16>>
  %input = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [8192, 128], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8192x128xf16>> -> tensor<8192x128xf16>
  %weight = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0], sizes = [128], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128xf16>> -> tensor<128xf16>
  %empty_2d = tensor.empty() : tensor<8192x128xf32>
  %empty_1d = tensor.empty() : tensor<8192xf32>
  %empty_out = tensor.empty() : tensor<8192x128xf16>

  // Op 1: extf + square (x^2 in f32)
  %squared = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%input : tensor<8192x128xf16>) outs(%empty_2d : tensor<8192x128xf32>) {
  ^bb0(%in: f16, %out: f32):
    %v = arith.extf %in : f16 to f32
    %sq = arith.mulf %v, %v : f32
    linalg.yield %sq : f32
  } -> tensor<8192x128xf32>

  // Op 2: fill for sum accumulator
  %acc_init = linalg.fill ins(%cst : f32) outs(%empty_1d : tensor<8192xf32>) -> tensor<8192xf32>

  // Op 3: reduce sum of squares along d1
  %sum_sq = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]}
    ins(%squared : tensor<8192x128xf32>) outs(%acc_init : tensor<8192xf32>) {
  ^bb0(%in: f32, %out: f32):
    %s = arith.addf %in, %out : f32
    linalg.yield %s : f32
  } -> tensor<8192xf32>

  // Op 4: mean + eps + rsqrt -> rrms = rsqrt(sum/N + eps)
  %cst_n = arith.constant 1.280000e+02 : f32
  %rrms = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>,
                     affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]}
    ins(%sum_sq : tensor<8192xf32>) outs(%empty_1d : tensor<8192xf32>) {
  ^bb0(%in: f32, %out: f32):
    %mean = arith.divf %in, %cst_n : f32
    %shifted = arith.addf %mean, %cst_eps : f32
    %r = math.rsqrt %shifted : f32
    linalg.yield %r : f32
  } -> tensor<8192xf32>

  // Op 5: normalize: truncf(extf(x) * rrms * extf(weight))
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>,
                     affine_map<(d0, d1) -> (d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%input, %rrms, %weight : tensor<8192x128xf16>, tensor<8192xf32>, tensor<128xf16>) outs(%empty_out : tensor<8192x128xf16>) {
  ^bb0(%in: f16, %scale: f32, %w: f16, %out: f16):
    %xf = arith.extf %in : f16 to f32
    %wf = arith.extf %w : f16 to f32
    %normed = arith.mulf %xf, %scale : f32
    %weighted = arith.mulf %normed, %wf : f32
    %r = arith.truncf %weighted : f32 to f16
    linalg.yield %r : f16
  } -> tensor<8192x128xf16>

  iree_tensor_ext.dispatch.tensor.store %result, %2, offsets = [0, 0], sizes = [8192, 128], strides = [1, 1] : tensor<8192x128xf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8192x128xf16>>
  return
}
