module {
  func.func @rms_norm_transpose_input(%arg0: tensor<4096x2048xf16>, %arg1: tensor<4096xf16>) -> (tensor<4096x2048xf16>) {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_eps = arith.constant 1.0e-06 : f32
    %cst_n = arith.constant 4096.0 : f32
    %c0 = arith.constant 0 : index
  
    // Reduction along d0 (NON-contiguous) for each d1 position
    %empty_rstd = tensor.empty() : tensor<2048xf32>
    %fill = linalg.fill ins(%cst : f32) outs(%empty_rstd : tensor<2048xf32>) -> tensor<2048xf32>
  
    // Op 1: Sum of squares along d0
    %sumsq = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d1)>],
      iterator_types = ["reduction", "parallel"]}
      ins(%arg0 : tensor<4096x2048xf16>) outs(%fill : tensor<2048xf32>) {
    ^bb0(%x: f16, %acc: f32):
      %xf = arith.extf %x : f16 to f32
      %sq = arith.mulf %xf, %xf : f32
      %add = arith.addf %sq, %acc : f32
      linalg.yield %add : f32
    } -> tensor<2048xf32>
  
    // Op 2: rsqrt(mean_sq + eps)
    %rstd = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%sumsq : tensor<2048xf32>) outs(%empty_rstd : tensor<2048xf32>) {
    ^bb0(%s: f32, %out: f32):
      %mean = arith.divf %s, %cst_n : f32
      %eps = arith.addf %mean, %cst_eps : f32
      %rs = math.rsqrt %eps : f32
      linalg.yield %rs : f32
    } -> tensor<2048xf32>
  
    // Op 3: Normalize: out[d0][d1] = in[d0][d1] * rstd[d1] * weight[d0]
    %empty_out = tensor.empty() : tensor<4096x2048xf16>
    %norm = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d1)>,
                       affine_map<(d0, d1) -> (d0)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg0, %rstd, %arg1 : tensor<4096x2048xf16>, tensor<2048xf32>, tensor<4096xf16>)
      outs(%empty_out : tensor<4096x2048xf16>) {
    ^bb0(%x: f16, %r: f32, %wt: f16, %out: f16):
      %xf = arith.extf %x : f16 to f32
      %normed = arith.mulf %xf, %r : f32
      %wf = arith.extf %wt : f16 to f32
      %scaled = arith.mulf %normed, %wf : f32
      %result_val = arith.truncf %scaled : f32 to f16
      linalg.yield %result_val : f16
    } -> tensor<4096x2048xf16>
  
    return %norm : tensor<4096x2048xf16>
  }
}
