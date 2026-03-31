module {
  func.func @reduction_3d(%arg0: tensor<8x256x4096xf32>) -> (tensor<8x4096xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %3 = tensor.empty() : tensor<8x4096xf32>
    %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<8x4096xf32>) -> tensor<8x4096xf32>
    %5 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d2)>],
      iterator_types = ["parallel", "reduction", "parallel"]}
      ins(%arg0 : tensor<8x256x4096xf32>) outs(%4 : tensor<8x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %6 = arith.addf %in, %out : f32
      linalg.yield %6 : f32
    } -> tensor<8x4096xf32>
    return %5 : tensor<8x4096xf32>
  }
}
