module {
  func.func @elementwise_unaligned(%arg0: tensor<1000x3xf32>) -> (tensor<1000x3xf32>) {
    %c0 = arith.constant 0 : index
  
  
  
    %empty = tensor.empty() : tensor<1000x3xf32>
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg0 : tensor<1000x3xf32>) outs(%empty : tensor<1000x3xf32>) {
    ^bb0(%in: f32, %out: f32):
      %abs = math.absf %in : f32
      linalg.yield %abs : f32
    } -> tensor<1000x3xf32>
  
    return %result : tensor<1000x3xf32>
  }
}
