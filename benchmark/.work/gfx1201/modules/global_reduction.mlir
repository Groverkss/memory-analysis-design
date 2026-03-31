module {
  func.func @global_reduction(%arg0: tensor<16777216xf32>) -> (tensor<f32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %3 = tensor.empty() : tensor<f32>
    %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<f32>) -> tensor<f32>
    %5 = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> ()>],
      iterator_types = ["reduction"]}
      ins(%arg0 : tensor<16777216xf32>) outs(%4 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %6 = arith.addf %in, %out : f32
      linalg.yield %6 : f32
    } -> tensor<f32>
    return %5 : tensor<f32>
  }
}
