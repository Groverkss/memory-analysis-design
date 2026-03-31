module {
  func.func @outer_reduction_f16(%arg0: tensor<4096x8192xf16>) -> (tensor<8192xf16>) {
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %3 = tensor.empty() : tensor<8192xf16>
    %4 = linalg.fill ins(%cst : f16) outs(%3 : tensor<8192xf16>) -> tensor<8192xf16>
    %5 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d1)>],
      iterator_types = ["reduction", "parallel"]}
      ins(%arg0 : tensor<4096x8192xf16>) outs(%4 : tensor<8192xf16>) {
    ^bb0(%in: f16, %out: f16):
      %6 = arith.addf %in, %out : f16
      linalg.yield %6 : f16
    } -> tensor<8192xf16>
    return %5 : tensor<8192xf16>
  }
}
