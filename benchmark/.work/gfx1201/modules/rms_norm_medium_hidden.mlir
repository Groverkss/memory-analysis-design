module {
  func.func @rms_norm_medium_hidden(%arg0: tensor<2048x4096xf16>, %arg1: tensor<4096xf16>) -> (tensor<2048x4096xf16>) {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_eps = arith.constant 9.99999974E-6 : f32
    %cst_n = arith.constant 4.096000e+03 : f32
    %c0 = arith.constant 0 : index
    %empty_2d = tensor.empty() : tensor<2048x4096xf32>
    %empty_1d = tensor.empty() : tensor<2048xf32>
    %empty_out = tensor.empty() : tensor<2048x4096xf16>
  
    // Op 1: extf + square
    %squared = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg0 : tensor<2048x4096xf16>) outs(%empty_2d : tensor<2048x4096xf32>) {
    ^bb0(%in: f16, %out: f32):
      %v = arith.extf %in : f16 to f32
      %sq = arith.mulf %v, %v : f32
      linalg.yield %sq : f32
    } -> tensor<2048x4096xf32>
  
    // Op 2: fill for reduction
    %acc_init = linalg.fill ins(%cst : f32) outs(%empty_1d : tensor<2048xf32>) -> tensor<2048xf32>
  
    // Op 3: reduce sum of squares along d1
    %sum_sq = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]}
      ins(%squared : tensor<2048x4096xf32>) outs(%acc_init : tensor<2048xf32>) {
    ^bb0(%in: f32, %out: f32):
      %s = arith.addf %in, %out : f32
      linalg.yield %s : f32
    } -> tensor<2048xf32>
  
    // Op 4: rsqrt(mean + eps)
    %rrms = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%sum_sq : tensor<2048xf32>) outs(%empty_1d : tensor<2048xf32>) {
    ^bb0(%in: f32, %out: f32):
      %mean = arith.divf %in, %cst_n : f32
      %shifted = arith.addf %mean, %cst_eps : f32
      %r = math.rsqrt %shifted : f32
      linalg.yield %r : f32
    } -> tensor<2048xf32>
  
    // Op 5: normalize + weight
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0)>,
                       affine_map<(d0, d1) -> (d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg0, %rrms, %arg1 : tensor<2048x4096xf16>, tensor<2048xf32>, tensor<4096xf16>) outs(%empty_out : tensor<2048x4096xf16>) {
    ^bb0(%in: f16, %scale: f32, %w: f16, %out: f16):
      %xf = arith.extf %in : f16 to f32
      %wf = arith.extf %w : f16 to f32
      %normed = arith.mulf %xf, %scale : f32
      %weighted = arith.mulf %normed, %wf : f32
      %r = arith.truncf %weighted : f32 to f16
      linalg.yield %r : f16
    } -> tensor<2048x4096xf16>
  
    return %result : tensor<2048x4096xf16>
  }
}
