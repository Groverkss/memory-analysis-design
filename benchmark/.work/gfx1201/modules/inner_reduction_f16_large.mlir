module {
  func.func @inner_reduction_f16_large(%arg0: tensor<1024x65536xf16>) -> (tensor<1024xf16>) {
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %3 = tensor.empty() : tensor<1024xf16>
    %4 = linalg.fill ins(%cst : f16) outs(%3 : tensor<1024xf16>) -> tensor<1024xf16>
    %5 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]}
      ins(%arg0 : tensor<1024x65536xf16>) outs(%4 : tensor<1024xf16>) {
    ^bb0(%in: f16, %out: f16):
      %6 = arith.addf %in, %out : f16
      linalg.yield %6 : f16
    } -> tensor<1024xf16>
    return %5 : tensor<1024xf16>
  }
}
