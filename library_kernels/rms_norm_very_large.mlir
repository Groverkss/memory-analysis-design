// RMSNorm with very large hidden dimension (65536), batch=64
// Pattern: Needs multi-block cooperation. Apex uses CTAS_PER_ROW>1,
// Quack uses cluster_n>1. One block of 256 threads cannot cover 65536
// elements efficiently in a single pass.
// tensor<64x65536xf16> -> tensor<64x65536xf16>, weights tensor<65536xf16>
//
// == Optimal Configuration ==
// hidden_dim=65536, batch=64, f16 input/output, f32 accumulation
// threads_per_row = 256 (max block size)
// vector_width = 8 (f16)
// elements_per_thread = 256 (65536 / 256 = 256, i.e., 32 vector loads!)
// rows_per_block = 1
// workgroup_size = 256
// num_workgroups = 64 (one per row, single-block approach)
//
// Multi-block alternative (Apex CTAS_PER_ROW=4):
//   4 blocks per row, 256 threads each. elements_per_block = 16384.
//   elements_per_thread = 64 (16384/256). Each block does a partial
//   reduction, then an atomic add or second kernel for final reduce.
//   num_workgroups = 64 * 4 = 256.
//
// Single-block analysis (what IREE currently does):
//   256 elements per thread = 32 vector loads of 8 f16.
//   reduction_strategy: 256 threads -> 8 warps, shuffle + shmem.
//   The reduction itself is fast; the bottleneck is the 32 loads.
//   coalescing: d1 innermost, 256*8*2 = 4096 bytes per vector pass.
//     32 passes to cover the row. Each pass is fully coalesced.
//   register pressure: 256 f16 inputs accumulated into 1 f32 partial sum
//     per thread. Only need to keep running sum, not all elements.
//     But second pass (normalize) needs to reload all 256 elements.
//   Low occupancy: only 64 workgroups for the whole GPU. On an SM80
//     with 108 SMs, most SMs are idle. This is where multi-block helps.

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @rms_norm_very_large() {
  %cst = arith.constant 0.000000e+00 : f32
  %cst_eps = arith.constant 9.99999974E-6 : f32
  %cst_n = arith.constant 6.553600e+04 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x65536xf16>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<65536xf16>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<64x65536xf16>>
  %input = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [64, 65536], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x65536xf16>> -> tensor<64x65536xf16>
  %weight = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0], sizes = [65536], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<65536xf16>> -> tensor<65536xf16>
  %empty_2d = tensor.empty() : tensor<64x65536xf32>
  %empty_1d = tensor.empty() : tensor<64xf32>
  %empty_out = tensor.empty() : tensor<64x65536xf16>

  // Op 1: extf + square
  %squared = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%input : tensor<64x65536xf16>) outs(%empty_2d : tensor<64x65536xf32>) {
  ^bb0(%in: f16, %out: f32):
    %v = arith.extf %in : f16 to f32
    %sq = arith.mulf %v, %v : f32
    linalg.yield %sq : f32
  } -> tensor<64x65536xf32>

  // Op 2: fill
  %acc_init = linalg.fill ins(%cst : f32) outs(%empty_1d : tensor<64xf32>) -> tensor<64xf32>

  // Op 3: reduce sum of squares
  %sum_sq = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]}
    ins(%squared : tensor<64x65536xf32>) outs(%acc_init : tensor<64xf32>) {
  ^bb0(%in: f32, %out: f32):
    %s = arith.addf %in, %out : f32
    linalg.yield %s : f32
  } -> tensor<64xf32>

  // Op 4: rsqrt(mean + eps)
  %rrms = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>,
                     affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]}
    ins(%sum_sq : tensor<64xf32>) outs(%empty_1d : tensor<64xf32>) {
  ^bb0(%in: f32, %out: f32):
    %mean = arith.divf %in, %cst_n : f32
    %shifted = arith.addf %mean, %cst_eps : f32
    %r = math.rsqrt %shifted : f32
    linalg.yield %r : f32
  } -> tensor<64xf32>

  // Op 5: normalize + weight
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>,
                     affine_map<(d0, d1) -> (d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%input, %rrms, %weight : tensor<64x65536xf16>, tensor<64xf32>, tensor<65536xf16>) outs(%empty_out : tensor<64x65536xf16>) {
  ^bb0(%in: f16, %scale: f32, %w: f16, %out: f16):
    %xf = arith.extf %in : f16 to f32
    %wf = arith.extf %w : f16 to f32
    %normed = arith.mulf %xf, %scale : f32
    %weighted = arith.mulf %normed, %wf : f32
    %r = arith.truncf %weighted : f32 to f16
    linalg.yield %r : f16
  } -> tensor<64x65536xf16>

  iree_tensor_ext.dispatch.tensor.store %result, %2, offsets = [0, 0], sizes = [64, 65536], strides = [1, 1] : tensor<64x65536xf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<64x65536xf16>>
  return
}
