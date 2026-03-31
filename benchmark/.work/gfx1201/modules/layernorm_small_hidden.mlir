module {
  func.func @layernorm_small_hidden(%arg0: tensor<32768x64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>) -> (tensor<32768x64xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_eps = arith.constant 9.99999974E-6 : f32
    %cst_n = arith.constant 6.400000e+01 : f32
    %c0 = arith.constant 0 : index
    %empty_2d = tensor.empty() : tensor<32768x64xf32>
    %empty_1d = tensor.empty() : tensor<32768xf32>
  
    // Op 1: fill for mean
    %acc_init = linalg.fill ins(%cst : f32) outs(%empty_1d : tensor<32768xf32>) -> tensor<32768xf32>
  
    // Op 2: sum for mean
    %sum = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]}
      ins(%arg0 : tensor<32768x64xf32>) outs(%acc_init : tensor<32768xf32>) {
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
      ins(%arg0, %mean : tensor<32768x64xf32>, tensor<32768xf32>) outs(%var_init : tensor<32768xf32>) {
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
      ins(%arg0, %mean, %rstd, %arg1, %arg2 : tensor<32768x64xf32>, tensor<32768xf32>, tensor<32768xf32>, tensor<64xf32>, tensor<64xf32>) outs(%empty_2d : tensor<32768x64xf32>) {
    ^bb0(%in: f32, %m: f32, %rs: f32, %g: f32, %b: f32, %out: f32):
      %diff = arith.subf %in, %m : f32
      %normed = arith.mulf %diff, %rs : f32
      %scaled = arith.mulf %normed, %g : f32
      %biased = arith.addf %scaled, %b : f32
      linalg.yield %biased : f32
    } -> tensor<32768x64xf32>
  
    return %result : tensor<32768x64xf32>
  }
}
