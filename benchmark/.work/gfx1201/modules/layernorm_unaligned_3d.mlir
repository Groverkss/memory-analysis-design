module {
  func.func @layernorm_unaligned_3d(%arg0: tensor<16x100x511xf16>, %arg1: tensor<511xf16>, %arg2: tensor<511xf16>) -> (tensor<16x100x511xf16>) {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_eps = arith.constant 9.99999974E-6 : f32
    %cst_n = arith.constant 5.110000e+02 : f32
    %c0 = arith.constant 0 : index
    %empty_3d = tensor.empty() : tensor<16x100x511xf32>
    %empty_2d = tensor.empty() : tensor<16x100xf32>
    %empty_out = tensor.empty() : tensor<16x100x511xf16>
  
    // Op 1: extf
    %extended = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%arg0 : tensor<16x100x511xf16>) outs(%empty_3d : tensor<16x100x511xf32>) {
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
      ins(%arg0, %mean, %rstd, %arg1, %arg2 : tensor<16x100x511xf16>, tensor<16x100xf32>, tensor<16x100xf32>, tensor<511xf16>, tensor<511xf16>) outs(%empty_out : tensor<16x100x511xf16>) {
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
  
    return %result : tensor<16x100x511xf16>
  }
}
