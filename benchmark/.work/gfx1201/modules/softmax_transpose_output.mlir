module {
  func.func @softmax_transpose_output(%arg0: tensor<1024x2048xf16>) -> (tensor<2048x1024xf16>) {
    %c0 = arith.constant 0 : index
    %cst_neg_inf = arith.constant 0xFF800000 : f16
    %cst_zero = arith.constant 0.000000e+00 : f16
  
    // Stage 1: max reduction along d1
    %empty_row = tensor.empty() : tensor<1024xf16>
    %fill_neg_inf = linalg.fill ins(%cst_neg_inf : f16) outs(%empty_row : tensor<1024xf16>) -> tensor<1024xf16>
    %max_val = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]}
      ins(%arg0 : tensor<1024x2048xf16>) outs(%fill_neg_inf : tensor<1024xf16>) {
    ^bb0(%a: f16, %b: f16):
      %v = arith.maximumf %a, %b : f16
      linalg.yield %v : f16
    } -> tensor<1024xf16>
  
    // Stage 2: subtract max + exp
    %empty_full = tensor.empty() : tensor<1024x2048xf16>
    %exp_val = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg0, %max_val : tensor<1024x2048xf16>, tensor<1024xf16>)
      outs(%empty_full : tensor<1024x2048xf16>) {
    ^bb0(%a: f16, %m: f16, %out: f16):
      %sub = arith.subf %a, %m : f16
      %e = math.exp %sub : f16
      linalg.yield %e : f16
    } -> tensor<1024x2048xf16>
  
    // Stage 3: sum reduction along d1
    %fill_zero = linalg.fill ins(%cst_zero : f16) outs(%empty_row : tensor<1024xf16>) -> tensor<1024xf16>
    %sum_val = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]}
      ins(%exp_val : tensor<1024x2048xf16>) outs(%fill_zero : tensor<1024xf16>) {
    ^bb0(%a: f16, %b: f16):
      %v = arith.addf %a, %b : f16
      linalg.yield %v : f16
    } -> tensor<1024xf16>
  
    // Stage 4: divide + TRANSPOSE OUTPUT
    // Output indexing: (d0, d1) -> (d1, d0) -- TRANSPOSED
    %empty_out = tensor.empty() : tensor<2048x1024xf16>
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0)>,
                       affine_map<(d0, d1) -> (d1, d0)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%exp_val, %sum_val : tensor<1024x2048xf16>, tensor<1024xf16>)
      outs(%empty_out : tensor<2048x1024xf16>) {
    ^bb0(%e: f16, %s: f16, %out: f16):
      %v = arith.divf %e, %s : f16
      linalg.yield %v : f16
    } -> tensor<2048x1024xf16>
  
    return %result : tensor<2048x1024xf16>
  }
}
