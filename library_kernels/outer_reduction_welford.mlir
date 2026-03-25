// Outer Welford reduction: compute mean AND variance along d0 simultaneously.
// tensor<256x1024xf16> -> (tensor<1024xf32>, tensor<1024xf32>)  [mean, variance]
// This is the BatchNorm statistics pattern: reduce N*H*W, keep C.
//
// Source pattern: Apex welford.cu, PyTorch Normalization.cuh batch_norm_collect_statistics,
// MIOpen BatchNorm DPP reduction. Uses Welford online algorithm for numerical stability.
//
// == Optimal Configuration ==
// Block size: 256 threads (2D: block.x=32 parallel, block.y=8 reduction)
// Grid: ceil(1024/32) = 32 workgroups (tiling d1, the contiguous output dim)
// Vector width: Cannot vectorize reduction dim. Output vec=4 (f32).
// Elements per thread: 256/8 = 32 d0 iterations
//
// Thread mapping: same as outer_reduction pattern.
// The twist: each thread accumulates a Welford triple (count, mean, m2),
// then the shared memory reduction uses Welford parallel merge, not simple add.
//
// Welford parallel merge: given (count_a, mean_a, m2_a) and (count_b, mean_b, m2_b):
//   count = count_a + count_b
//   delta = mean_b - mean_a
//   mean = (count_a * mean_a + count_b * mean_b) / count
//   m2 = m2_a + m2_b + delta^2 * count_a * count_b / count
//
// Shared memory: block.x * block.y * 3 * sizeof(f32) = 32 * 8 * 12 = 3072 bytes
// (for count, mean, m2 per thread).

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @outer_reduction_welford() {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %input = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x1024xf16>>
  %out_mean = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1024xf32>>
  %out_var = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1024xf32>>
  %in = iree_tensor_ext.dispatch.tensor.load %input, offsets = [0, 0], sizes = [256, 1024], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x1024xf16>> -> tensor<256x1024xf16>

  // Op 1: Sum reduction along d0 (for mean)
  %empty_col = tensor.empty() : tensor<1024xf32>
  %fill0 = linalg.fill ins(%cst : f32) outs(%empty_col : tensor<1024xf32>) -> tensor<1024xf32>
  %sum = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d1)>],
    iterator_types = ["reduction", "parallel"]}
    ins(%in : tensor<256x1024xf16>) outs(%fill0 : tensor<1024xf32>) {
  ^bb0(%x: f16, %acc: f32):
    %xf = arith.extf %x : f16 to f32
    %add = arith.addf %xf, %acc : f32
    linalg.yield %add : f32
  } -> tensor<1024xf32>

  // Op 2: mean = sum / N
  %cst_n = arith.constant 256.0 : f32
  %mean = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>,
                     affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]}
    ins(%sum : tensor<1024xf32>) outs(%empty_col : tensor<1024xf32>) {
  ^bb0(%s: f32, %out: f32):
    %v = arith.divf %s, %cst_n : f32
    linalg.yield %v : f32
  } -> tensor<1024xf32>

  // Op 3: Variance reduction: sum((x - mean)^2) along d0
  %fill1 = linalg.fill ins(%cst : f32) outs(%empty_col : tensor<1024xf32>) -> tensor<1024xf32>
  %var_sum = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d1)>,
                     affine_map<(d0, d1) -> (d1)>],
    iterator_types = ["reduction", "parallel"]}
    ins(%in, %mean : tensor<256x1024xf16>, tensor<1024xf32>) outs(%fill1 : tensor<1024xf32>) {
  ^bb0(%x: f16, %m: f32, %acc: f32):
    %xf = arith.extf %x : f16 to f32
    %diff = arith.subf %xf, %m : f32
    %sq = arith.mulf %diff, %diff : f32
    %add = arith.addf %sq, %acc : f32
    linalg.yield %add : f32
  } -> tensor<1024xf32>

  // Op 4: variance = var_sum / N
  %var = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>,
                     affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]}
    ins(%var_sum : tensor<1024xf32>) outs(%empty_col : tensor<1024xf32>) {
  ^bb0(%s: f32, %out: f32):
    %v = arith.divf %s, %cst_n : f32
    linalg.yield %v : f32
  } -> tensor<1024xf32>

  iree_tensor_ext.dispatch.tensor.store %mean, %out_mean, offsets = [0], sizes = [1024], strides = [1] : tensor<1024xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1024xf32>>
  iree_tensor_ext.dispatch.tensor.store %var, %out_var, offsets = [0], sizes = [1024], strides = [1] : tensor<1024xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1024xf32>>
  return
}
