module {
  func.func @layernorm_transpose_output(%arg0: tensor<2048x4096xf16>, %arg1: tensor<4096xf16>) -> (tensor<4096x2048xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_n = arith.constant 4096.0 : f32
    %cst_eps = arith.constant 1.0e-05 : f32
    %c0 = arith.constant 0 : index
    %empty_row = tensor.empty() : tensor<2048xf32>
    %empty_full = tensor.empty() : tensor<2048x4096xf32>
    %empty_out = tensor.empty() : tensor<4096x2048xf32>
  
    // Op 1: extf input to f32
    %ext = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg0 : tensor<2048x4096xf16>) outs(%empty_full : tensor<2048x4096xf32>) {
    ^bb0(%a: f16, %b: f32):
      %v = arith.extf %a : f16 to f32
      linalg.yield %v : f32
    } -> tensor<2048x4096xf32>
  
    // Op 2: mean reduction
    %fill0 = linalg.fill ins(%cst : f32) outs(%empty_row : tensor<2048xf32>) -> tensor<2048xf32>
    %mean_sum = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]}
      ins(%ext : tensor<2048x4096xf32>) outs(%fill0 : tensor<2048xf32>) {
    ^bb0(%a: f32, %b: f32):
      %v = arith.addf %a, %b : f32
      linalg.yield %v : f32
    } -> tensor<2048xf32>
  
    // Op 3: mean = sum / N
    %mean = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%mean_sum : tensor<2048xf32>) outs(%empty_row : tensor<2048xf32>) {
    ^bb0(%a: f32, %b: f32):
      %v = arith.divf %a, %cst_n : f32
      linalg.yield %v : f32
    } -> tensor<2048xf32>
  
    // Op 4: variance reduction
    %fill1 = linalg.fill ins(%cst : f32) outs(%empty_row : tensor<2048xf32>) -> tensor<2048xf32>
    %var_sum = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0)>,
                       affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]}
      ins(%ext, %mean : tensor<2048x4096xf32>, tensor<2048xf32>) outs(%fill1 : tensor<2048xf32>) {
    ^bb0(%x: f32, %m: f32, %acc: f32):
      %sub = arith.subf %x, %m : f32
      %sq = arith.mulf %sub, %sub : f32
      %add = arith.addf %sq, %acc : f32
      linalg.yield %add : f32
    } -> tensor<2048xf32>
  
    // Op 5: normalize + weight + TRANSPOSE OUTPUT
    // Input indexing: (d0, d1) -> normal
    // Output indexing: (d0, d1) -> (d1, d0)  -- TRANSPOSED
    %norm = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0)>,
                       affine_map<(d0, d1) -> (d0)>,
                       affine_map<(d0, d1) -> (d1)>,
                       affine_map<(d0, d1) -> (d1, d0)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%ext, %mean, %var_sum, %arg1 : tensor<2048x4096xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<4096xf16>)
      outs(%empty_out : tensor<4096x2048xf32>) {
    ^bb0(%x: f32, %m: f32, %v: f32, %wt: f16, %out: f32):
      %var = arith.divf %v, %cst_n : f32
      %eps = arith.addf %var, %cst_eps : f32
      %rstd = math.rsqrt %eps : f32
      %centered = arith.subf %x, %m : f32
      %normed = arith.mulf %centered, %rstd : f32
      %wf = arith.extf %wt : f16 to f32
      %scaled = arith.mulf %normed, %wf : f32
      linalg.yield %scaled : f32
    } -> tensor<4096x2048xf32>
  
    return %norm : tensor<4096x2048xf32>
  }
}
