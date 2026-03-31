module {
  func.func @transpose_4d_bsnh_to_bnsh(%arg0: tensor<4x2048x32x128xf16>) -> (tensor<4x32x2048x128xf16>) {
    %c0 = arith.constant 0 : index
    %3 = tensor.empty() : tensor<4x32x2048x128xf16>
    // d0=batch, d1=seq, d2=heads, d3=headdim
    // input: (d0, d1, d2, d3), output: (d0, d2, d1, d3)
    %4 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%arg0 : tensor<4x2048x32x128xf16>) outs(%3 : tensor<4x32x2048x128xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<4x32x2048x128xf16>
    return %4 : tensor<4x32x2048x128xf16>
  }
}
