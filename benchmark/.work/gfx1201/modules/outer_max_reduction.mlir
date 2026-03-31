module {
  func.func @outer_max_reduction(%arg0: tensor<2048x512xf32>) -> (tensor<512xf32>) {
    %cst = arith.constant 0xFF800000 : f32
    %c0 = arith.constant 0 : index
    %3 = tensor.empty() : tensor<512xf32>
    %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<512xf32>) -> tensor<512xf32>
    %5 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d1)>],
      iterator_types = ["reduction", "parallel"]}
      ins(%arg0 : tensor<2048x512xf32>) outs(%4 : tensor<512xf32>) {
    ^bb0(%in: f32, %out: f32):
      %6 = arith.maximumf %in, %out : f32
      linalg.yield %6 : f32
    } -> tensor<512xf32>
    return %5 : tensor<512xf32>
  }
}
