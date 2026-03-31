module {
  func.func @cross_entropy_fused(%arg0: tensor<2048x32000xf16>) -> (tensor<2048xf16>) {
    %cst_neg_inf = arith.constant 0xFC00 : f16
    %cst_zero = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
  
    // Step 1: Max reduction along d1.
    %empty_max = tensor.empty() : tensor<2048xf16>
    %fill_max = linalg.fill ins(%cst_neg_inf : f16) outs(%empty_max : tensor<2048xf16>) -> tensor<2048xf16>
    %row_max = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]}
      ins(%arg0 : tensor<2048x32000xf16>) outs(%fill_max : tensor<2048xf16>) {
    ^bb0(%in: f16, %out: f16):
      %v = arith.maximumf %in, %out : f16
      linalg.yield %v : f16
    } -> tensor<2048xf16>
  
    // Step 2: Subtract max and exponentiate (elementwise, fused).
    %empty_exp = tensor.empty() : tensor<2048x32000xf16>
    %exp_shifted = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg0, %row_max : tensor<2048x32000xf16>, tensor<2048xf16>)
      outs(%empty_exp : tensor<2048x32000xf16>) {
    ^bb0(%x: f16, %mx: f16, %out: f16):
      %shifted = arith.subf %x, %mx : f16
      %exp_val = math.exp %shifted : f16
      linalg.yield %exp_val : f16
    } -> tensor<2048x32000xf16>
  
    // Step 3: Sum reduction of exp values along d1.
    %empty_sum = tensor.empty() : tensor<2048xf16>
    %fill_sum = linalg.fill ins(%cst_zero : f16) outs(%empty_sum : tensor<2048xf16>) -> tensor<2048xf16>
    %row_sum_exp = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]}
      ins(%exp_shifted : tensor<2048x32000xf16>) outs(%fill_sum : tensor<2048xf16>) {
    ^bb0(%in: f16, %out: f16):
      %v = arith.addf %in, %out : f16
      linalg.yield %v : f16
    } -> tensor<2048xf16>
  
    // Step 4: log(sum_exp) + max = logsumexp (elementwise).
    %empty_result = tensor.empty() : tensor<2048xf16>
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%row_sum_exp, %row_max : tensor<2048xf16>, tensor<2048xf16>)
      outs(%empty_result : tensor<2048xf16>) {
    ^bb0(%sum_val: f16, %max_val: f16, %out: f16):
      %log_val = math.log %sum_val : f16
      %lse = arith.addf %log_val, %max_val : f16
      linalg.yield %lse : f16
    } -> tensor<2048xf16>
  
    return %result : tensor<2048xf16>
  }
}
