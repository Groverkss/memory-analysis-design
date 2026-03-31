module {
  func.func @geglu_fused(%arg0: tensor<2048x4096xf16>, %arg1: tensor<2048x4096xf16>) -> (tensor<2048x4096xf16>) {
    %c0 = arith.constant 0 : index
    %cst_half = arith.constant 0.500000e+00 : f32
    %cst_one = arith.constant 1.000000e+00 : f32
    %cst_coeff = arith.constant 0.044715 : f32
    // sqrt(2/pi) = 0.7978845608...
    %cst_sqrt2pi = arith.constant 0.7978845608 : f32
  
  
  
    %empty = tensor.empty() : tensor<2048x4096xf16>
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg0, %arg1 : tensor<2048x4096xf16>, tensor<2048x4096xf16>) outs(%empty : tensor<2048x4096xf16>) {
    ^bb0(%gate_val: f16, %up_val: f16, %out: f16):
      // geglu(gate, up) = gelu_tanh(gate) * up
      // gelu_tanh(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
      %g = arith.extf %gate_val : f16 to f32
      %u = arith.extf %up_val : f16 to f32
      %g2 = arith.mulf %g, %g : f32
      %g3 = arith.mulf %g2, %g : f32
      %coeff_g3 = arith.mulf %cst_coeff, %g3 : f32
      %inner = arith.addf %g, %coeff_g3 : f32
      %scaled = arith.mulf %cst_sqrt2pi, %inner : f32
      %tanh_val = math.tanh %scaled : f32
      %one_plus_tanh = arith.addf %cst_one, %tanh_val : f32
      %half_x = arith.mulf %cst_half, %g : f32
      %gelu = arith.mulf %half_x, %one_plus_tanh : f32
      %geglu = arith.mulf %gelu, %u : f32
      %result = arith.truncf %geglu : f32 to f16
      linalg.yield %result : f16
    } -> tensor<2048x4096xf16>
  
    return %result : tensor<2048x4096xf16>
  }
}
