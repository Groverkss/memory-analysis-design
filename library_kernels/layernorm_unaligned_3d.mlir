// LayerNorm over tensor<16x100x511xf16>, reduce over d2 (size 511)
// Pattern: Non-power-of-2 reduction dim (511). Tests masking with odd sizes.
// 511 is prime, so no clean factorization. Tests worst-case alignment.
//
// == Optimal Configuration ==
// reduction_size = 511, parallel_size = 16 * 100 = 1600
// f16 input/output, f32 accumulation
// threads_per_row = 64 (ceil(511/8) = 63.875 -> 64 threads, which is a pow2)
//   64 threads * 8 vec_elems = 512 capacity. 511 elements means the last
//   thread's last vector element is masked (512 - 511 = 1 element masked).
// vector_width = 8 (f16)
// elements_per_thread = 8 (511/64 ~ 7.98, rounded up to 8 with masking)
// rows_per_block = 4 (256 / 64 = 4 rows per block)
// workgroup_size = 256
// num_workgroups = 400 (1600 / 4)
// reduction_strategy: 64 threads -> 2 warps per row.
//   5 shuffle rounds within warp, then 1 shared memory round for 2 partials.
// coalescing: d2 innermost, 64 * 8 * 2 = 1024 bytes per row (but only
//   511 * 2 = 1022 bytes valid). 4 rows = 4088 bytes per WG.
// masking: 511 elements with 512 thread*vec capacity. Only 1 element
//   is masked out (thread 63's last vec element). Very efficient.
//   However, 511 is not a multiple of 8 (vec width), so the last
//   vector load needs a partial/masked load.

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @layernorm_unaligned_3d() {
  %cst = arith.constant 0.000000e+00 : f32
  %cst_eps = arith.constant 9.99999974E-6 : f32
  %cst_n = arith.constant 5.110000e+02 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16x100x511xf16>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<511xf16>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<511xf16>>
  %3 = hal.interface.binding.subspan layout(#pipeline_layout) binding(3) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16x100x511xf16>>
  %input = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [16, 100, 511], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16x100x511xf16>> -> tensor<16x100x511xf16>
  %gamma = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0], sizes = [511], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<511xf16>> -> tensor<511xf16>
  %beta = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0], sizes = [511], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<511xf16>> -> tensor<511xf16>
  %empty_3d = tensor.empty() : tensor<16x100x511xf32>
  %empty_2d = tensor.empty() : tensor<16x100xf32>
  %empty_out = tensor.empty() : tensor<16x100x511xf16>

  // Op 1: extf
  %extended = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                     affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
    iterator_types = ["parallel", "parallel", "parallel"]}
    ins(%input : tensor<16x100x511xf16>) outs(%empty_3d : tensor<16x100x511xf32>) {
  ^bb0(%in: f16, %out: f32):
    %v = arith.extf %in : f16 to f32
    linalg.yield %v : f32
  } -> tensor<16x100x511xf32>

  // Op 2: fill for mean
  %acc_init = linalg.fill ins(%cst : f32) outs(%empty_2d : tensor<16x100xf32>) -> tensor<16x100xf32>

  // Op 3: sum over d2 for mean
  %sum = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                     affine_map<(d0, d1, d2) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel", "reduction"]}
    ins(%extended : tensor<16x100x511xf32>) outs(%acc_init : tensor<16x100xf32>) {
  ^bb0(%in: f32, %out: f32):
    %s = arith.addf %in, %out : f32
    linalg.yield %s : f32
  } -> tensor<16x100xf32>

  // Op 4: mean = sum / N
  %mean = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%sum : tensor<16x100xf32>) outs(%empty_2d : tensor<16x100xf32>) {
  ^bb0(%in: f32, %out: f32):
    %m = arith.divf %in, %cst_n : f32
    linalg.yield %m : f32
  } -> tensor<16x100xf32>

  // Op 5: fill for variance
  %var_init = linalg.fill ins(%cst : f32) outs(%empty_2d : tensor<16x100xf32>) -> tensor<16x100xf32>

  // Op 6: variance reduction over d2
  %var_sum = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                     affine_map<(d0, d1, d2) -> (d0, d1)>,
                     affine_map<(d0, d1, d2) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel", "reduction"]}
    ins(%extended, %mean : tensor<16x100x511xf32>, tensor<16x100xf32>) outs(%var_init : tensor<16x100xf32>) {
  ^bb0(%in: f32, %m: f32, %out: f32):
    %diff = arith.subf %in, %m : f32
    %sq = arith.mulf %diff, %diff : f32
    %s = arith.addf %sq, %out : f32
    linalg.yield %s : f32
  } -> tensor<16x100xf32>

  // Op 7: rstd = rsqrt(var/N + eps)
  %rstd = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%var_sum : tensor<16x100xf32>) outs(%empty_2d : tensor<16x100xf32>) {
  ^bb0(%in: f32, %out: f32):
    %v = arith.divf %in, %cst_n : f32
    %shifted = arith.addf %v, %cst_eps : f32
    %r = math.rsqrt %shifted : f32
    linalg.yield %r : f32
  } -> tensor<16x100xf32>

  // Op 8: normalize: (x - mean) * rstd * gamma + beta
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                     affine_map<(d0, d1, d2) -> (d0, d1)>,
                     affine_map<(d0, d1, d2) -> (d0, d1)>,
                     affine_map<(d0, d1, d2) -> (d2)>,
                     affine_map<(d0, d1, d2) -> (d2)>,
                     affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
    iterator_types = ["parallel", "parallel", "parallel"]}
    ins(%input, %mean, %rstd, %gamma, %beta : tensor<16x100x511xf16>, tensor<16x100xf32>, tensor<16x100xf32>, tensor<511xf16>, tensor<511xf16>) outs(%empty_out : tensor<16x100x511xf16>) {
  ^bb0(%in: f16, %m: f32, %rs: f32, %g: f16, %b: f16, %out: f16):
    %xf = arith.extf %in : f16 to f32
    %gf = arith.extf %g : f16 to f32
    %bf = arith.extf %b : f16 to f32
    %diff = arith.subf %xf, %m : f32
    %normed = arith.mulf %diff, %rs : f32
    %scaled = arith.mulf %normed, %gf : f32
    %biased = arith.addf %scaled, %bf : f32
    %r = arith.truncf %biased : f32 to f16
    linalg.yield %r : f16
  } -> tensor<16x100x511xf16>

  iree_tensor_ext.dispatch.tensor.store %result, %3, offsets = [0, 0, 0], sizes = [16, 100, 511], strides = [1, 1, 1] : tensor<16x100x511xf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16x100x511xf16>>
  return
}
