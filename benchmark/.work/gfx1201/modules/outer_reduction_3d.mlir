module {
  func.func @outer_reduction_3d(%arg0: tensor<256x32x4096xf16>) -> (tensor<32x4096xf16>) {
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %3 = tensor.empty() : tensor<32x4096xf16>
    %4 = linalg.fill ins(%cst : f16) outs(%3 : tensor<32x4096xf16>) -> tensor<32x4096xf16>
    %5 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d1, d2)>],
      iterator_types = ["reduction", "parallel", "parallel"]}
      ins(%arg0 : tensor<256x32x4096xf16>) outs(%4 : tensor<32x4096xf16>) {
    ^bb0(%in: f16, %out: f16):
      %6 = arith.addf %in, %out : f16
      linalg.yield %6 : f16
    } -> tensor<32x4096xf16>
    return %5 : tensor<32x4096xf16>
  }
}
