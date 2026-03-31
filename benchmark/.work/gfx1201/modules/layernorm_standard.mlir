module {
  func.func @layernorm_standard(%arg0: tensor<2048x4096xf16>, %arg1: tensor<4096xf16>, %arg2: tensor<4096xf16>) -> (tensor<2048x4096xf16>) {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_eps = arith.constant 9.99999974E-6 : f32
    %cst_n = arith.constant 4.096000e+03 : f32
    %c0 = arith.constant 0 : index
    %empty_2d = tensor.empty() : tensor<2048x4096xf32>
    %empty_1d = tensor.empty() : tensor<2048xf32>
    %empty_out = tensor.empty() : tensor<2048x4096xf16>
  
    // Op 1: extf (f16 -> f32)
    %extended = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg0 : tensor<2048x4096xf16>) outs(%empty_2d : tensor<2048x4096xf32>) {
    ^bb0(%in: f16, %out: f32):
      %v = arith.extf %in : f16 to f32
      linalg.yield %v : f32
    } -> tensor<2048x4096xf32>
  
    // Op 2: fill for mean accumulator
    %acc_init = linalg.fill ins(%cst : f32) outs(%empty_1d : tensor<2048xf32>) -> tensor<2048xf32>
  
    // Op 3: sum reduction for mean
    %sum = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]}
      ins(%extended : tensor<2048x4096xf32>) outs(%acc_init : tensor<2048xf32>) {
    ^bb0(%in: f32, %out: f32):
      %s = arith.addf %in, %out : f32
      linalg.yield %s : f32
    } -> tensor<2048xf32>
  
    // Op 4: divide to get mean
    %mean = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%sum : tensor<2048xf32>) outs(%empty_1d : tensor<2048xf32>) {
    ^bb0(%in: f32, %out: f32):
      %m = arith.divf %in, %cst_n : f32
      linalg.yield %m : f32
    } -> tensor<2048xf32>
  
    // Op 5: fill for variance accumulator (reuse zero init)
    %var_init = linalg.fill ins(%cst : f32) outs(%empty_1d : tensor<2048xf32>) -> tensor<2048xf32>
  
    // Op 6: variance reduction: sum((x - mean)^2)
    %var_sum = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0)>,
                       affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]}
      ins(%extended, %mean : tensor<2048x4096xf32>, tensor<2048xf32>) outs(%var_init : tensor<2048xf32>) {
    ^bb0(%in: f32, %m: f32, %out: f32):
      %diff = arith.subf %in, %m : f32
      %sq = arith.mulf %diff, %diff : f32
      %s = arith.addf %sq, %out : f32
      linalg.yield %s : f32
    } -> tensor<2048xf32>
  
    // Op 7: variance -> rstd = rsqrt(var/N + eps)
    %rstd = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%var_sum : tensor<2048xf32>) outs(%empty_1d : tensor<2048xf32>) {
    ^bb0(%in: f32, %out: f32):
      %v = arith.divf %in, %cst_n : f32
      %shifted = arith.addf %v, %cst_eps : f32
      %r = math.rsqrt %shifted : f32
      linalg.yield %r : f32
    } -> tensor<2048xf32>
  
    // Op 8: normalize: (x - mean) * rstd * gamma + beta
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0)>,
                       affine_map<(d0, d1) -> (d0)>,
                       affine_map<(d0, d1) -> (d1)>,
                       affine_map<(d0, d1) -> (d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg0, %mean, %rstd, %arg2, %arg1 : tensor<2048x4096xf16>, tensor<2048xf32>, tensor<2048xf32>, tensor<4096xf16>, tensor<4096xf16>) outs(%empty_out : tensor<2048x4096xf16>) {
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
    } -> tensor<2048x4096xf16>
  
    return %result : tensor<2048x4096xf16>
  }
}
