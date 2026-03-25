// LayerNorm with very small hidden dimension (64), large batch (32768)
// Pattern: Per-head normalization (QKRMSNorm from FlashInfer). Single warp suffices.
// tensor<32768x64xf32> -> tensor<32768x64xf32>, gamma/beta tensor<64xf32>
// All f32, no type conversion needed.
//
// == Optimal Configuration ==
// hidden_dim=64, batch=32768, f32 throughout
// threads_per_row = 16 (64 / 4 vec_elems = 16 threads per row)
//   4 vec_elems because f32: 128 bits / 32 bits = 4
// vector_width = 4 (128 bits / 32 bits)
// elements_per_thread = 4 (64 / 16 = 4, exactly one vector load)
// rows_per_block = 16 (256 / 16 = 16)
// workgroup_size = 256
// num_workgroups = 2048 (32768 / 16)
// reduction_strategy: 16 threads per row -> single half-warp.
//   4 shuffle rounds (log2(16)=4). No shared memory needed.
//   This is the lightest possible reduction config.
// coalescing: d1 innermost, 16 * 4 * 4 = 256 bytes per row.
//   16 rows = 4096 bytes per WG. Good coalescing.
// register pressure: 4 f32 per thread per pass = 16 bytes. Trivial.

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @layernorm_small_hidden() {
  %cst = arith.constant 0.000000e+00 : f32
  %cst_eps = arith.constant 9.99999974E-6 : f32
  %cst_n = arith.constant 6.400000e+01 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32768x64xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64xf32>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64xf32>>
  %3 = hal.interface.binding.subspan layout(#pipeline_layout) binding(3) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32768x64xf32>>
  %input = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [32768, 64], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32768x64xf32>> -> tensor<32768x64xf32>
  %gamma = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0], sizes = [64], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64xf32>> -> tensor<64xf32>
  %beta = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0], sizes = [64], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64xf32>> -> tensor<64xf32>
  %empty_2d = tensor.empty() : tensor<32768x64xf32>
  %empty_1d = tensor.empty() : tensor<32768xf32>

  // Op 1: fill for mean
  %acc_init = linalg.fill ins(%cst : f32) outs(%empty_1d : tensor<32768xf32>) -> tensor<32768xf32>

  // Op 2: sum for mean
  %sum = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]}
    ins(%input : tensor<32768x64xf32>) outs(%acc_init : tensor<32768xf32>) {
  ^bb0(%in: f32, %out: f32):
    %s = arith.addf %in, %out : f32
    linalg.yield %s : f32
  } -> tensor<32768xf32>

  // Op 3: mean = sum / N
  %mean = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>,
                     affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]}
    ins(%sum : tensor<32768xf32>) outs(%empty_1d : tensor<32768xf32>) {
  ^bb0(%in: f32, %out: f32):
    %m = arith.divf %in, %cst_n : f32
    linalg.yield %m : f32
  } -> tensor<32768xf32>

  // Op 4: fill for variance
  %var_init = linalg.fill ins(%cst : f32) outs(%empty_1d : tensor<32768xf32>) -> tensor<32768xf32>

  // Op 5: variance = sum((x - mean)^2)
  %var_sum = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>,
                     affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]}
    ins(%input, %mean : tensor<32768x64xf32>, tensor<32768xf32>) outs(%var_init : tensor<32768xf32>) {
  ^bb0(%in: f32, %m: f32, %out: f32):
    %diff = arith.subf %in, %m : f32
    %sq = arith.mulf %diff, %diff : f32
    %s = arith.addf %sq, %out : f32
    linalg.yield %s : f32
  } -> tensor<32768xf32>

  // Op 6: rstd = rsqrt(var/N + eps)
  %rstd = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>,
                     affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]}
    ins(%var_sum : tensor<32768xf32>) outs(%empty_1d : tensor<32768xf32>) {
  ^bb0(%in: f32, %out: f32):
    %v = arith.divf %in, %cst_n : f32
    %shifted = arith.addf %v, %cst_eps : f32
    %r = math.rsqrt %shifted : f32
    linalg.yield %r : f32
  } -> tensor<32768xf32>

  // Op 7: normalize: (x - mean) * rstd * gamma + beta
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>,
                     affine_map<(d0, d1) -> (d0)>,
                     affine_map<(d0, d1) -> (d1)>,
                     affine_map<(d0, d1) -> (d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%input, %mean, %rstd, %gamma, %beta : tensor<32768x64xf32>, tensor<32768xf32>, tensor<32768xf32>, tensor<64xf32>, tensor<64xf32>) outs(%empty_2d : tensor<32768x64xf32>) {
  ^bb0(%in: f32, %m: f32, %rs: f32, %g: f32, %b: f32, %out: f32):
    %diff = arith.subf %in, %m : f32
    %normed = arith.mulf %diff, %rs : f32
    %scaled = arith.mulf %normed, %g : f32
    %biased = arith.addf %scaled, %b : f32
    linalg.yield %biased : f32
  } -> tensor<32768x64xf32>

  iree_tensor_ext.dispatch.tensor.store %result, %3, offsets = [0, 0], sizes = [32768, 64], strides = [1, 1] : tensor<32768x64xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32768x64xf32>>
  return
}
