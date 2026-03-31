module {
  func.func @elementwise_3d_broadcast(%arg0: tensor<8x512x4096xf16>, %arg1: tensor<4096xf16>) -> (tensor<8x512x4096xf16>) {
    %c0 = arith.constant 0 : index
  
  
  
    %empty = tensor.empty() : tensor<8x512x4096xf16>
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%arg0, %arg1 : tensor<8x512x4096xf16>, tensor<4096xf16>) outs(%empty : tensor<8x512x4096xf16>) {
    ^bb0(%in: f16, %b: f16, %out: f16):
      %in_f32 = arith.extf %in : f16 to f32
      %b_f32 = arith.extf %b : f16 to f32
      %sum = arith.addf %in_f32, %b_f32 : f32
      %result = arith.truncf %sum : f32 to f16
      linalg.yield %result : f16
    } -> tensor<8x512x4096xf16>
  
    return %result : tensor<8x512x4096xf16>
  }
}
