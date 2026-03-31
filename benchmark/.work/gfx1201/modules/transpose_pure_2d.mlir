module {
  func.func @transpose_pure_2d(%arg0: tensor<2048x4096xf16>) -> (tensor<4096x2048xf16>) {
    %c0 = arith.constant 0 : index
    %3 = tensor.empty() : tensor<4096x2048xf16>
    %4 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d1, d0)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg0 : tensor<2048x4096xf16>) outs(%3 : tensor<4096x2048xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<4096x2048xf16>
    return %4 : tensor<4096x2048xf16>
  }
}
