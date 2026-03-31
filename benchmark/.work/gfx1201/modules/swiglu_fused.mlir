module {
  func.func @swiglu_fused(%arg0: tensor<2048x4096xf16>, %arg1: tensor<2048x4096xf16>) -> (tensor<2048x4096xf16>) {
    %c0 = arith.constant 0 : index
    %cst_one = arith.constant 1.000000e+00 : f32
  
  
  
    %empty = tensor.empty() : tensor<2048x4096xf16>
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg0, %arg1 : tensor<2048x4096xf16>, tensor<2048x4096xf16>) outs(%empty : tensor<2048x4096xf16>) {
    ^bb0(%gate_val: f16, %up_val: f16, %out: f16):
      // swiglu(gate, up) = silu(gate) * up = gate * sigmoid(gate) * up
      %g = arith.extf %gate_val : f16 to f32
      %u = arith.extf %up_val : f16 to f32
      %neg = arith.negf %g : f32
      %exp = math.exp %neg : f32
      %denom = arith.addf %exp, %cst_one : f32
      %sigmoid = arith.divf %cst_one, %denom : f32
      %silu = arith.mulf %g, %sigmoid : f32
      %swiglu = arith.mulf %silu, %u : f32
      %result = arith.truncf %swiglu : f32 to f16
      linalg.yield %result : f16
    } -> tensor<2048x4096xf16>
  
    return %result : tensor<2048x4096xf16>
  }
}
