module {
  func.func @transpose_3d_inner(%arg0: tensor<64x512x1024xf32>) -> (tensor<64x1024x512xf32>) {
    %c0 = arith.constant 0 : index
    %3 = tensor.empty() : tensor<64x1024x512xf32>
    %4 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d2, d1)>],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%arg0 : tensor<64x512x1024xf32>) outs(%3 : tensor<64x1024x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<64x1024x512xf32>
    return %4 : tensor<64x1024x512xf32>
  }
}
