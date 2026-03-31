module {
  func.func @rope_embedding(%arg0: tensor<4x2048x32x128xf16>, %arg1: tensor<2048x64xf16>, %arg2: tensor<2048x64xf16>) -> (tensor<4x2048x32x128xf16>) {
    %c0 = arith.constant 0 : index
  
    // Input: [batch=4, seq=2048, heads=32, head_dim=128]
    // Cos table: [seq=2048, half_head=64], broadcast over batch and heads
    // Sin table: [seq=2048, half_head=64], broadcast over batch and heads
  
  
    // Extract first half [0:64] and second half [64:128] of head_dim.
    %first_half = tensor.extract_slice %arg0[0, 0, 0, 0] [4, 2048, 32, 64] [1, 1, 1, 1] : tensor<4x2048x32x128xf16> to tensor<4x2048x32x64xf16>
    %second_half = tensor.extract_slice %arg0[0, 0, 0, 64] [4, 2048, 32, 64] [1, 1, 1, 1] : tensor<4x2048x32x128xf16> to tensor<4x2048x32x64xf16>
  
    // Compute rotated first half: x1 * cos - x2 * sin
    %empty_first = tensor.empty() : tensor<4x2048x32x64xf16>
    %rotated_first = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d1, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d1, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%first_half, %second_half, %arg1, %arg2 : tensor<4x2048x32x64xf16>, tensor<4x2048x32x64xf16>, tensor<2048x64xf16>, tensor<2048x64xf16>) outs(%empty_first : tensor<4x2048x32x64xf16>) {
    ^bb0(%x1: f16, %x2: f16, %c: f16, %s: f16, %out: f16):
      %x1_f32 = arith.extf %x1 : f16 to f32
      %x2_f32 = arith.extf %x2 : f16 to f32
      %c_f32 = arith.extf %c : f16 to f32
      %s_f32 = arith.extf %s : f16 to f32
      %a = arith.mulf %x1_f32, %c_f32 : f32
      %b = arith.mulf %x2_f32, %s_f32 : f32
      %r = arith.subf %a, %b : f32
      %result = arith.truncf %r : f32 to f16
      linalg.yield %result : f16
    } -> tensor<4x2048x32x64xf16>
  
    // Compute rotated second half: x2 * cos + x1 * sin
    %empty_second = tensor.empty() : tensor<4x2048x32x64xf16>
    %rotated_second = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d1, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d1, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%second_half, %first_half, %arg1, %arg2 : tensor<4x2048x32x64xf16>, tensor<4x2048x32x64xf16>, tensor<2048x64xf16>, tensor<2048x64xf16>) outs(%empty_second : tensor<4x2048x32x64xf16>) {
    ^bb0(%x2: f16, %x1: f16, %c: f16, %s: f16, %out: f16):
      %x2_f32 = arith.extf %x2 : f16 to f32
      %x1_f32 = arith.extf %x1 : f16 to f32
      %c_f32 = arith.extf %c : f16 to f32
      %s_f32 = arith.extf %s : f16 to f32
      %a = arith.mulf %x2_f32, %c_f32 : f32
      %b = arith.mulf %x1_f32, %s_f32 : f32
      %r = arith.addf %a, %b : f32
      %result = arith.truncf %r : f32 to f16
      linalg.yield %result : f16
    } -> tensor<4x2048x32x64xf16>
  
    // Concatenate the two halves back into full head_dim.
    %empty_out = tensor.empty() : tensor<4x2048x32x128xf16>
    %inserted_first = tensor.insert_slice %rotated_first into %empty_out[0, 0, 0, 0] [4, 2048, 32, 64] [1, 1, 1, 1] : tensor<4x2048x32x64xf16> into tensor<4x2048x32x128xf16>
    %output = tensor.insert_slice %rotated_second into %inserted_first[0, 0, 0, 64] [4, 2048, 32, 64] [1, 1, 1, 1] : tensor<4x2048x32x64xf16> into tensor<4x2048x32x128xf16>
  
    return %output : tensor<4x2048x32x128xf16>
  }
}
