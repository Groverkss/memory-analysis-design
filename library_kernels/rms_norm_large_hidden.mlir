// RMSNorm with large hidden dimension (16384), batch=512
// Pattern: GPT-4 scale model. All 256 threads per row, multiple vector loads.
// tensor<512x16384xbf16> -> tensor<512x16384xbf16>, weights tensor<16384xbf16>
// Computation: x * rsqrt(mean(x^2) + eps) * weight
//
// == Optimal Configuration ==
// hidden_dim=16384, batch=512, bf16 input/output, f32 accumulation
// threads_per_row = 256 (needs full block for one row)
// vector_width = 8 (128 bits / 16 bits = 8 bf16 elements)
// elements_per_thread = 64 (16384 / 256 = 64, i.e., 8 vector loads)
// rows_per_block = 1
// workgroup_size = 256
// num_workgroups = 512 (one per row)
// reduction_strategy: same as medium (256 threads -> 8 warps -> shuffle + shmem)
//   but more work per thread before the reduction (64 elements to accumulate).
// coalescing: d1 innermost, 256 * 8 * 2 = 4096 bytes per vector pass.
//   8 passes to cover the row. Each pass is fully coalesced.
// register pressure: 64 bf16 inputs + 64 f32 intermediates per thread
//   = 128 + 256 = 384 bytes. Moderate, but within register file budget.
// data fits in registers: 64 elements per thread. Two passes through data.

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @rms_norm_large_hidden() {
  %cst = arith.constant 0.000000e+00 : f32
  %cst_eps = arith.constant 9.99999974E-6 : f32
  %cst_n = arith.constant 1.638400e+04 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<512x16384xbf16>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16384xbf16>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<512x16384xbf16>>
  %input = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [512, 16384], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<512x16384xbf16>> -> tensor<512x16384xbf16>
  %weight = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0], sizes = [16384], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16384xbf16>> -> tensor<16384xbf16>
  %empty_2d = tensor.empty() : tensor<512x16384xf32>
  %empty_1d = tensor.empty() : tensor<512xf32>
  %empty_out = tensor.empty() : tensor<512x16384xbf16>

  // Op 1: extf + square
  %squared = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%input : tensor<512x16384xbf16>) outs(%empty_2d : tensor<512x16384xf32>) {
  ^bb0(%in: bf16, %out: f32):
    %v = arith.extf %in : bf16 to f32
    %sq = arith.mulf %v, %v : f32
    linalg.yield %sq : f32
  } -> tensor<512x16384xf32>

  // Op 2: fill
  %acc_init = linalg.fill ins(%cst : f32) outs(%empty_1d : tensor<512xf32>) -> tensor<512xf32>

  // Op 3: reduce sum of squares
  %sum_sq = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]}
    ins(%squared : tensor<512x16384xf32>) outs(%acc_init : tensor<512xf32>) {
  ^bb0(%in: f32, %out: f32):
    %s = arith.addf %in, %out : f32
    linalg.yield %s : f32
  } -> tensor<512xf32>

  // Op 4: rsqrt(mean + eps)
  %rrms = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>,
                     affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]}
    ins(%sum_sq : tensor<512xf32>) outs(%empty_1d : tensor<512xf32>) {
  ^bb0(%in: f32, %out: f32):
    %mean = arith.divf %in, %cst_n : f32
    %shifted = arith.addf %mean, %cst_eps : f32
    %r = math.rsqrt %shifted : f32
    linalg.yield %r : f32
  } -> tensor<512xf32>

  // Op 5: normalize + weight
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>,
                     affine_map<(d0, d1) -> (d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%input, %rrms, %weight : tensor<512x16384xbf16>, tensor<512xf32>, tensor<16384xbf16>) outs(%empty_out : tensor<512x16384xbf16>) {
  ^bb0(%in: bf16, %scale: f32, %w: bf16, %out: bf16):
    %xf = arith.extf %in : bf16 to f32
    %wf = arith.extf %w : bf16 to f32
    %normed = arith.mulf %xf, %scale : f32
    %weighted = arith.mulf %normed, %wf : f32
    %r = arith.truncf %weighted : f32 to bf16
    linalg.yield %r : bf16
  } -> tensor<512x16384xbf16>

  iree_tensor_ext.dispatch.tensor.store %result, %2, offsets = [0, 0], sizes = [512, 16384], strides = [1, 1] : tensor<512x16384xbf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<512x16384xbf16>>
  return
}
