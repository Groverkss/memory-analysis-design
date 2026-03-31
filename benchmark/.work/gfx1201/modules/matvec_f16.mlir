module {
  func.func @matvec_f16(%arg0: tensor<4096x4096xf16>, %arg1: tensor<4096xf16>) -> (tensor<4096xf16>) {
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %3 = tensor.empty() : tensor<4096xf16>
    %4 = linalg.fill ins(%cst : f16) outs(%3 : tensor<4096xf16>) -> tensor<4096xf16>
    %5 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d1)>,
                       affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]}
      ins(%arg0, %arg1 : tensor<4096x4096xf16>, tensor<4096xf16>)
      outs(%4 : tensor<4096xf16>) {
    ^bb0(%a: f16, %x: f16, %out: f16):
      %mul = arith.mulf %a, %x : f16
      %add = arith.addf %mul, %out : f16
      linalg.yield %add : f16
    } -> tensor<4096xf16>
    return %5 : tensor<4096xf16>
  }
}
