module {
  func.func @outer_reduction_welford(%arg0: tensor<256x1024xf16>) -> (tensor<1024xf32>, tensor<1024xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
  
    // Op 1: Sum reduction along d0 (for mean)
    %empty_col = tensor.empty() : tensor<1024xf32>
    %fill0 = linalg.fill ins(%cst : f32) outs(%empty_col : tensor<1024xf32>) -> tensor<1024xf32>
    %sum = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d1)>],
      iterator_types = ["reduction", "parallel"]}
      ins(%arg0 : tensor<256x1024xf16>) outs(%fill0 : tensor<1024xf32>) {
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
      ins(%arg0, %mean : tensor<256x1024xf16>, tensor<1024xf32>) outs(%fill1 : tensor<1024xf32>) {
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
  
    return %mean, %var : tensor<1024xf32>, tensor<1024xf32>
  }
}
