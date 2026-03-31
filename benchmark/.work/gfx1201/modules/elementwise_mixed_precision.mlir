module {
  func.func @elementwise_mixed_precision(%arg0: tensor<4096x4096xf8E4M3FNUZ>, %arg1: tensor<128x128xf32>) -> (tensor<4096x4096xf16>) {
    %c0 = arith.constant 0 : index
  
  
  
    %empty = tensor.empty() : tensor<4096x4096xf16>
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0 floordiv 32, d1 floordiv 32)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg0, %arg1 : tensor<4096x4096xf8E4M3FNUZ>, tensor<128x128xf32>) outs(%empty : tensor<4096x4096xf16>) {
    ^bb0(%in: f8E4M3FNUZ, %s: f32, %out: f16):
      %val = arith.extf %in : f8E4M3FNUZ to f32
      %scaled = arith.mulf %val, %s : f32
      %result = arith.truncf %scaled : f32 to f16
      linalg.yield %result : f16
    } -> tensor<4096x4096xf16>
  
    return %result : tensor<4096x4096xf16>
  }
}
