module {
  func.func @argmax_row(%arg0: tensor<2048x32000xf32>) -> (tensor<2048xi64>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32000 = arith.constant 32000 : index
    %c2048 = arith.constant 2048 : index
    %cst_neg_inf = arith.constant 0xFF800000 : f32
    %cst_zero_idx = arith.constant 0 : i64
    %3 = tensor.empty() : tensor<2048xi64>
    %4 = tensor.empty() : tensor<2048xf32>
    // Fill with -inf for max value, 0 for index.
    %5 = linalg.fill ins(%cst_neg_inf : f32) outs(%4 : tensor<2048xf32>) -> tensor<2048xf32>
    %6 = linalg.fill ins(%cst_zero_idx : i64) outs(%3 : tensor<2048xi64>) -> tensor<2048xi64>
    // Reduction carrying (max_value, max_index) pair.
    %7:2 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0)>,
                       affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]}
      ins(%arg0 : tensor<2048x32000xf32>)
      outs(%5, %6 : tensor<2048xf32>, tensor<2048xi64>) {
    ^bb0(%in: f32, %out_val: f32, %out_idx: i64):
      %idx = linalg.index 1 : index
      %idx_i64 = arith.index_cast %idx : index to i64
      %cmp = arith.cmpf ogt, %in, %out_val : f32
      %new_val = arith.select %cmp, %in, %out_val : f32
      %new_idx = arith.select %cmp, %idx_i64, %out_idx : i64
      linalg.yield %new_val, %new_idx : f32, i64
    } -> (tensor<2048xf32>, tensor<2048xi64>)
    return %7#1 : tensor<2048xi64>
  }
}
