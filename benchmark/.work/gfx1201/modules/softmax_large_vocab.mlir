module {
  func.func @softmax_large_vocab(%arg0: tensor<4096x32000xf16>) -> (tensor<4096x32000xf16>) {
    %cst_min = arith.constant 0xFC00 : f16
    %cst_zero = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
  
  
  
    // Op1: Max reduction along d1.
    %empty_max = tensor.empty() : tensor<4096xf16>
    %fill_max = linalg.fill ins(%cst_min : f16) outs(%empty_max : tensor<4096xf16>) -> tensor<4096xf16>
    %max = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]}
      ins(%arg0 : tensor<4096x32000xf16>) outs(%fill_max : tensor<4096xf16>) {
    ^bb0(%in: f16, %out: f16):
      %0 = arith.maximumf %in, %out : f16
      linalg.yield %0 : f16
    } -> tensor<4096xf16>
  
    // Op2: Subtract max (broadcast) + exp.
    %empty_exp = tensor.empty() : tensor<4096x32000xf16>
    %exp = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg0, %max : tensor<4096x32000xf16>, tensor<4096xf16>) outs(%empty_exp : tensor<4096x32000xf16>) {
    ^bb0(%in: f16, %mx: f16, %out: f16):
      %0 = arith.subf %in, %mx : f16
      %1 = math.exp %0 : f16
      linalg.yield %1 : f16
    } -> tensor<4096x32000xf16>
  
    // Op3: Sum reduction along d1.
    %empty_sum = tensor.empty() : tensor<4096xf16>
    %fill_sum = linalg.fill ins(%cst_zero : f16) outs(%empty_sum : tensor<4096xf16>) -> tensor<4096xf16>
    %sum = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]}
      ins(%exp : tensor<4096x32000xf16>) outs(%fill_sum : tensor<4096xf16>) {
    ^bb0(%in: f16, %out: f16):
      %0 = arith.addf %in, %out : f16
      linalg.yield %0 : f16
    } -> tensor<4096xf16>
  
    // Op4: Divide by sum (broadcast).
    %empty_out = tensor.empty() : tensor<4096x32000xf16>
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%exp, %sum : tensor<4096x32000xf16>, tensor<4096xf16>) outs(%empty_out : tensor<4096x32000xf16>) {
    ^bb0(%in: f16, %s: f16, %out: f16):
      %0 = arith.divf %in, %s : f16
      linalg.yield %0 : f16
    } -> tensor<4096x32000xf16>
  
    return %result : tensor<4096x32000xf16>
  }
}
