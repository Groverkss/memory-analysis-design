// RMSNorm with unaligned hidden dimension (768), BERT-base style
// Pattern: 768 is NOT a power of 2. Tests masking for threads that have no work.
// tensor<4096x768xf16> -> tensor<4096x768xf16>, weights tensor<768xf16>
// Computation: x * rsqrt(mean(x^2) + eps) * weight
//
// == Optimal Configuration ==
// hidden_dim=768, batch=4096, f16 input/output, f32 accumulation
// threads_per_row = 128 (next power of 2 >= ceil(768/8) = 96)
//   96 threads would suffice but must be power-of-2 for warp shuffle.
//   128 is the smallest valid threads_per_row.
// vector_width = 8 (128 bits / 16 bits)
// elements_per_thread = 8 (768 / 96 active threads = 8, but 128 threads
//   means 32 threads are idle/masked). Alternatively: ceil(768/128)=6
//   with masking on the last vector.
// rows_per_block = 2 (256 / 128 = 2 rows per block)
// workgroup_size = 256
// num_workgroups = 2048 (4096 / 2)
// reduction_strategy: 128 threads_per_row -> 4 warps per row.
//   5 shuffle rounds within warp, then shared memory for 4 partial sums.
// coalescing: d1 innermost, 128 * 8 * 2 = 2048 bytes per load, but only
//   768 * 2 = 1536 bytes are valid. Masking needed for last 256 bytes.
// masking: 768 = 96 * 8. With 128 threads, threads 96-127 produce zero
//   (masked out). The reduction still works because 0 is the identity.

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @rms_norm_unaligned() {
  %cst = arith.constant 0.000000e+00 : f32
  %cst_eps = arith.constant 9.99999974E-6 : f32
  %cst_n = arith.constant 7.680000e+02 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x768xf16>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<768xf16>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4096x768xf16>>
  %input = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [4096, 768], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x768xf16>> -> tensor<4096x768xf16>
  %weight = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0], sizes = [768], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<768xf16>> -> tensor<768xf16>
  %empty_2d = tensor.empty() : tensor<4096x768xf32>
  %empty_1d = tensor.empty() : tensor<4096xf32>
  %empty_out = tensor.empty() : tensor<4096x768xf16>

  // Op 1: extf + square
  %squared = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%input : tensor<4096x768xf16>) outs(%empty_2d : tensor<4096x768xf32>) {
  ^bb0(%in: f16, %out: f32):
    %v = arith.extf %in : f16 to f32
    %sq = arith.mulf %v, %v : f32
    linalg.yield %sq : f32
  } -> tensor<4096x768xf32>

  // Op 2: fill
  %acc_init = linalg.fill ins(%cst : f32) outs(%empty_1d : tensor<4096xf32>) -> tensor<4096xf32>

  // Op 3: reduce sum of squares
  %sum_sq = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]}
    ins(%squared : tensor<4096x768xf32>) outs(%acc_init : tensor<4096xf32>) {
  ^bb0(%in: f32, %out: f32):
    %s = arith.addf %in, %out : f32
    linalg.yield %s : f32
  } -> tensor<4096xf32>

  // Op 4: rsqrt(mean + eps)
  %rrms = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>,
                     affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]}
    ins(%sum_sq : tensor<4096xf32>) outs(%empty_1d : tensor<4096xf32>) {
  ^bb0(%in: f32, %out: f32):
    %mean = arith.divf %in, %cst_n : f32
    %shifted = arith.addf %mean, %cst_eps : f32
    %r = math.rsqrt %shifted : f32
    linalg.yield %r : f32
  } -> tensor<4096xf32>

  // Op 5: normalize + weight
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>,
                     affine_map<(d0, d1) -> (d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%input, %rrms, %weight : tensor<4096x768xf16>, tensor<4096xf32>, tensor<768xf16>) outs(%empty_out : tensor<4096x768xf16>) {
  ^bb0(%in: f16, %scale: f32, %w: f16, %out: f16):
    %xf = arith.extf %in : f16 to f32
    %wf = arith.extf %w : f16 to f32
    %normed = arith.mulf %xf, %scale : f32
    %weighted = arith.mulf %normed, %wf : f32
    %r = arith.truncf %weighted : f32 to f16
    linalg.yield %r : f16
  } -> tensor<4096x768xf16>

  iree_tensor_ext.dispatch.tensor.store %result, %2, offsets = [0, 0], sizes = [4096, 768], strides = [1, 1] : tensor<4096x768xf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4096x768xf16>>
  return
}
