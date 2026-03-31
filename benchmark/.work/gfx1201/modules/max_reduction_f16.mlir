module {
  func.func @max_reduction_f16(%arg0: tensor<4096x8192xf16>) -> (tensor<4096xf16>) {
    %cst_neg_inf = arith.constant 0xFC00 : f16
    %c0 = arith.constant 0 : index
    %3 = tensor.empty() : tensor<4096xf16>
    %4 = linalg.fill ins(%cst_neg_inf : f16) outs(%3 : tensor<4096xf16>) -> tensor<4096xf16>
    %5 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]}
      ins(%arg0 : tensor<4096x8192xf16>) outs(%4 : tensor<4096xf16>) {
    ^bb0(%in: f16, %out: f16):
      %6 = arith.maximumf %in, %out : f16
      linalg.yield %6 : f16
    } -> tensor<4096xf16>
    return %5 : tensor<4096xf16>
  }
}
