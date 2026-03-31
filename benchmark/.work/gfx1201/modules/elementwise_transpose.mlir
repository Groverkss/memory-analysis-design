module {
  func.func @elementwise_transpose(%arg0: tensor<1024x4096xf16>) -> (tensor<4096x1024xf32>) {
    %c0 = arith.constant 0 : index
    %cst_scale = arith.constant 0.0078125 : f32
  
  
  
    %empty = tensor.empty() : tensor<4096x1024xf32>
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg0 : tensor<1024x4096xf16>) outs(%empty : tensor<4096x1024xf32>) {
    ^bb0(%in: f16, %out: f32):
      %val = arith.extf %in : f16 to f32
      %scaled = arith.mulf %val, %cst_scale : f32
      linalg.yield %scaled : f32
    } -> tensor<4096x1024xf32>
  
    return %result : tensor<4096x1024xf32>
  }
}
