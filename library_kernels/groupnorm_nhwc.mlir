// GroupNorm: tensor<4x64x32x32xf32>, 8 groups, NHWC-style layout
// Pattern: MIOpen/Apex GroupNorm. Reduce over spatial + channels_per_group.
// N=4, C=64, H=32, W=32. 8 groups -> 8 channels per group.
// Reshape conceptually as tensor<4x8x8x32x32> and reduce over (c_per_g, H, W).
// Reduction size = 8 * 32 * 32 = 8192 elements per (N, group).
// Output: tensor<4x64x32x32xf32>, gamma tensor<64xf32>, beta tensor<64xf32>
//
// == Optimal Configuration ==
// reduction_size = 8192 (8 channels_per_group * 32 * 32 spatial)
// parallel_size = 4 * 8 = 32 (N * groups)
// f32 throughout
// threads_per_row = 256 (8192 / 4 = 2048 vec_elems to cover, needs full block)
// vector_width = 4 (128 bits / 32 bits)
// elements_per_thread = 32 (8192 / 256 = 32, i.e., 8 vector loads)
// rows_per_block = 1
// workgroup_size = 256
// num_workgroups = 32 (4 batches * 8 groups)
// reduction_strategy: 256 threads -> 8 warps, shuffle + shmem.
// coalescing: In NCHW layout, W=32 is innermost. 32 elements * 4 bytes = 128
//   bytes. One warp (32 threads) covers one row of W. But we reduce over
//   H and W and channels_per_group, so the innermost contiguous access
//   is W. With 256 threads, first 32 cover W, next 32 cover next H row, etc.
// register pressure: 32 f32 per thread = 128 bytes. Moderate.
//
// Note: We model this as a 5D tensor<4x8x8x32x32xf32> with d0,d1 parallel
// and d2,d3,d4 reduction. The output has gamma/beta indexed by the full
// channel (d1*8+d2), which we model via a collapsed channel dim.

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @groupnorm_nhwc() {
  %cst = arith.constant 0.000000e+00 : f32
  %cst_eps = arith.constant 9.99999974E-6 : f32
  %cst_n = arith.constant 8.192000e+03 : f32
  %c0 = arith.constant 0 : index
  // Input as 5D: [N=4, G=8, C/G=8, H=32, W=32]
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x8x8x32x32xf32>>
  // Gamma and beta: [C=64] but we index as [G=8, C/G=8]
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x8xf32>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x8xf32>>
  %3 = hal.interface.binding.subspan layout(#pipeline_layout) binding(3) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x8x8x32x32xf32>>
  %input = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0, 0, 0], sizes = [4, 8, 8, 32, 32], strides = [1, 1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x8x8x32x32xf32>> -> tensor<4x8x8x32x32xf32>
  %gamma = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [8, 8], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x8xf32>> -> tensor<8x8xf32>
  %beta = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0, 0], sizes = [8, 8], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x8xf32>> -> tensor<8x8xf32>
  %empty_5d = tensor.empty() : tensor<4x8x8x32x32xf32>
  %empty_2d = tensor.empty() : tensor<4x8xf32>

  // Op 1: fill for mean
  %acc_init = linalg.fill ins(%cst : f32) outs(%empty_2d : tensor<4x8xf32>) -> tensor<4x8xf32>

  // Op 2: sum over d2, d3, d4 (channels_per_group, H, W)
  %sum = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel", "reduction", "reduction", "reduction"]}
    ins(%input : tensor<4x8x8x32x32xf32>) outs(%acc_init : tensor<4x8xf32>) {
  ^bb0(%in: f32, %out: f32):
    %s = arith.addf %in, %out : f32
    linalg.yield %s : f32
  } -> tensor<4x8xf32>

  // Op 3: mean = sum / N
  %mean = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%sum : tensor<4x8xf32>) outs(%empty_2d : tensor<4x8xf32>) {
  ^bb0(%in: f32, %out: f32):
    %m = arith.divf %in, %cst_n : f32
    linalg.yield %m : f32
  } -> tensor<4x8xf32>

  // Op 4: fill for variance
  %var_init = linalg.fill ins(%cst : f32) outs(%empty_2d : tensor<4x8xf32>) -> tensor<4x8xf32>

  // Op 5: variance reduction
  %var_sum = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d1)>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel", "reduction", "reduction", "reduction"]}
    ins(%input, %mean : tensor<4x8x8x32x32xf32>, tensor<4x8xf32>) outs(%var_init : tensor<4x8xf32>) {
  ^bb0(%in: f32, %m: f32, %out: f32):
    %diff = arith.subf %in, %m : f32
    %sq = arith.mulf %diff, %diff : f32
    %s = arith.addf %sq, %out : f32
    linalg.yield %s : f32
  } -> tensor<4x8xf32>

  // Op 6: rstd = rsqrt(var/N + eps)
  %rstd = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%var_sum : tensor<4x8xf32>) outs(%empty_2d : tensor<4x8xf32>) {
  ^bb0(%in: f32, %out: f32):
    %v = arith.divf %in, %cst_n : f32
    %shifted = arith.addf %v, %cst_eps : f32
    %r = math.rsqrt %shifted : f32
    linalg.yield %r : f32
  } -> tensor<4x8xf32>

  // Op 7: normalize: (x - mean) * rstd * gamma + beta
  // gamma/beta are per-channel, indexed by (d1, d2) = (group, channel_in_group)
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d1)>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d1)>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d1, d2)>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d1, d2)>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]}
    ins(%input, %mean, %rstd, %gamma, %beta : tensor<4x8x8x32x32xf32>, tensor<4x8xf32>, tensor<4x8xf32>, tensor<8x8xf32>, tensor<8x8xf32>) outs(%empty_5d : tensor<4x8x8x32x32xf32>) {
  ^bb0(%in: f32, %m: f32, %rs: f32, %g: f32, %b: f32, %out: f32):
    %diff = arith.subf %in, %m : f32
    %normed = arith.mulf %diff, %rs : f32
    %scaled = arith.mulf %normed, %g : f32
    %biased = arith.addf %scaled, %b : f32
    linalg.yield %biased : f32
  } -> tensor<4x8x8x32x32xf32>

  iree_tensor_ext.dispatch.tensor.store %result, %3, offsets = [0, 0, 0, 0, 0], sizes = [4, 8, 8, 32, 32], strides = [1, 1, 1, 1, 1] : tensor<4x8x8x32x32xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x8x8x32x32xf32>>
  return
}
