module {
  func.func @elementwise_broadcast(%arg0: tensor<2048x4096xf32>, %arg1: tensor<4096xf32>) -> (tensor<2048x4096xf32>) {
    %c0 = arith.constant 0 : index
  
  
  
    %empty = tensor.empty() : tensor<2048x4096xf32>
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg0, %arg1 : tensor<2048x4096xf32>, tensor<4096xf32>) outs(%empty : tensor<2048x4096xf32>) {
    ^bb0(%in: f32, %b: f32, %out: f32):
      %sum = arith.addf %in, %b : f32
      linalg.yield %sum : f32
    } -> tensor<2048x4096xf32>
  
    return %result : tensor<2048x4096xf32>
  }
}
