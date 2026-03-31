module {
  func.func @elementwise_large_1d(%arg0: tensor<16777216xf16>) -> (tensor<16777216xf16>) {
    %c0 = arith.constant 0 : index
    %cst_scale = arith.constant 0.0078125 : f16
  
  
  
    %empty = tensor.empty() : tensor<16777216xf16>
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%arg0 : tensor<16777216xf16>) outs(%empty : tensor<16777216xf16>) {
    ^bb0(%in: f16, %out: f16):
      %scaled = arith.mulf %in, %cst_scale : f16
      linalg.yield %scaled : f16
    } -> tensor<16777216xf16>
  
    return %result : tensor<16777216xf16>
  }
}
