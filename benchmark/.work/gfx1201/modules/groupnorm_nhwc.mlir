module {
  func.func @groupnorm_nhwc(%arg0: tensor<4x8x8x32x32xf32>, %arg1: tensor<8x8xf32>, %arg2: tensor<8x8xf32>) -> (tensor<4x8x8x32x32xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_eps = arith.constant 9.99999974E-6 : f32
    %cst_n = arith.constant 8.192000e+03 : f32
    %c0 = arith.constant 0 : index
    // Input as 5D: [N=4, G=8, C/G=8, H=32, W=32]
    // Gamma and beta: [C=64] but we index as [G=8, C/G=8]
    %empty_5d = tensor.empty() : tensor<4x8x8x32x32xf32>
    %empty_2d = tensor.empty() : tensor<4x8xf32>
  
    // Op 1: fill for mean
    %acc_init = linalg.fill ins(%cst : f32) outs(%empty_2d : tensor<4x8xf32>) -> tensor<4x8xf32>
  
    // Op 2: sum over d2, d3, d4 (channels_per_group, H, W)
    %sum = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>,
                       affine_map<(d0, d1, d2, d3, d4) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel", "reduction", "reduction", "reduction"]}
      ins(%arg0 : tensor<4x8x8x32x32xf32>) outs(%acc_init : tensor<4x8xf32>) {
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
      ins(%arg0, %mean : tensor<4x8x8x32x32xf32>, tensor<4x8xf32>) outs(%var_init : tensor<4x8xf32>) {
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
      ins(%arg0, %mean, %rstd, %arg1, %arg2 : tensor<4x8x8x32x32xf32>, tensor<4x8xf32>, tensor<4x8xf32>, tensor<8x8xf32>, tensor<8x8xf32>) outs(%empty_5d : tensor<4x8x8x32x32xf32>) {
    ^bb0(%in: f32, %m: f32, %rs: f32, %g: f32, %b: f32, %out: f32):
      %diff = arith.subf %in, %m : f32
      %normed = arith.mulf %diff, %rs : f32
      %scaled = arith.mulf %normed, %g : f32
      %biased = arith.addf %scaled, %b : f32
      linalg.yield %biased : f32
    } -> tensor<4x8x8x32x32xf32>
  
    return %result : tensor<4x8x8x32x32xf32>
  }
}
