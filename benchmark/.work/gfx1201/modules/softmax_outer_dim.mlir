module {
  func.func @softmax_outer_dim(%arg0: tensor<4096x1024xf32>) -> (tensor<4096x1024xf32>) {
    %cst_min = arith.constant 0xFF800000 : f32
    %cst_zero = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
  
  
  
    // Op1: Max reduction along d0 (outer dim).
    %empty_max = tensor.empty() : tensor<1024xf32>
    %fill_max = linalg.fill ins(%cst_min : f32) outs(%empty_max : tensor<1024xf32>) -> tensor<1024xf32>
    %max = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d1)>],
      iterator_types = ["reduction", "parallel"]}
      ins(%arg0 : tensor<4096x1024xf32>) outs(%fill_max : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %0 = arith.maximumf %in, %out : f32
      linalg.yield %0 : f32
    } -> tensor<1024xf32>
  
    // Op2: Subtract max (broadcast along d0) + exp.
    %empty_exp = tensor.empty() : tensor<4096x1024xf32>
    %exp = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg0, %max : tensor<4096x1024xf32>, tensor<1024xf32>) outs(%empty_exp : tensor<4096x1024xf32>) {
    ^bb0(%in: f32, %mx: f32, %out: f32):
      %0 = arith.subf %in, %mx : f32
      %1 = math.exp %0 : f32
      linalg.yield %1 : f32
    } -> tensor<4096x1024xf32>
  
    // Op3: Sum reduction along d0 (outer dim).
    %empty_sum = tensor.empty() : tensor<1024xf32>
    %fill_sum = linalg.fill ins(%cst_zero : f32) outs(%empty_sum : tensor<1024xf32>) -> tensor<1024xf32>
    %sum = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d1)>],
      iterator_types = ["reduction", "parallel"]}
      ins(%exp : tensor<4096x1024xf32>) outs(%fill_sum : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %0 = arith.addf %in, %out : f32
      linalg.yield %0 : f32
    } -> tensor<1024xf32>
  
    // Op4: Divide by sum (broadcast along d0).
    %empty_out = tensor.empty() : tensor<4096x1024xf32>
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%exp, %sum : tensor<4096x1024xf32>, tensor<1024xf32>) outs(%empty_out : tensor<4096x1024xf32>) {
    ^bb0(%in: f32, %s: f32, %out: f32):
      %0 = arith.divf %in, %s : f32
      linalg.yield %0 : f32
    } -> tensor<4096x1024xf32>
  
    return %result : tensor<4096x1024xf32>
  }
}
