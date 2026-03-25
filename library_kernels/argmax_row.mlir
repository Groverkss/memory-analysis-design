// argmax_row.mlir
// Argmax along d1: tensor<2048x32000xf32> -> tensor<2048xi64>
// Pattern: llama.cpp argmax for token sampling. Reduction carries both max value
// AND its index. Uses arith.cmpf + arith.select to implement the argmax combiner.
//
// == Optimal Configuration ==
// Block size: 512 threads (1D). Balances occupancy and register pressure for
//   carrying two values (max_val, max_idx) per thread.
// Grid: 2048 blocks (one per row).
// Vector width: 4 (f32 vec4 = 16 bytes for the input loads).
//   Index computation is scalar (derived from thread ID + loop counter).
// Elements per thread: 32000 / 512 = 62.5 -> 63 elements/thread (last thread masks).
//   With vec=4: ~16 vectorized iterations per thread.
//   CUB MemBoundScaling: items_per_thread = 63 * 4 / sizeof(f32) = 63 (f32, no scaling).
// Reduction strategy: Warp shuffle reduction carrying (value, index) pairs.
//   Each thread loops over its ~63 elements, maintaining local (max_val, max_idx).
//   Warp shuffle: compare values, select winner's index. 512/32 = 16 warps.
//   16 warp winners reduced via smem (store both val and idx, 16 * 2 * 4 = 128 bytes).
// Argmax combiner logic:
//   cmp = arith.cmpf ogt, new_val, current_max  (new > current?)
//   max_val = arith.select cmp, new_val, current_max
//   max_idx = arith.select cmp, new_idx, current_idx
//   Tie-breaking: ogt means equal values keep the earlier (lower) index.
// Coalescing: thread k reads [row, k], [row, k+512], ... -> perfectly coalesced.
// Note: 32000 is not a power of 2 (32000 = 2^5 * 1000). Last iteration needs masking.
//   32000 / 512 = 62.5, so some threads process 62 elements, others 63.

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @argmax_row() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c32000 = arith.constant 32000 : index
  %c2048 = arith.constant 2048 : index
  %cst_neg_inf = arith.constant 0xFF800000 : f32
  %cst_zero_idx = arith.constant 0 : i64
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048x32000xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2048xi64>>
  %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [2048, 32000], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048x32000xf32>> -> tensor<2048x32000xf32>
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
    ins(%2 : tensor<2048x32000xf32>)
    outs(%5, %6 : tensor<2048xf32>, tensor<2048xi64>) {
  ^bb0(%in: f32, %out_val: f32, %out_idx: i64):
    %idx = linalg.index 1 : index
    %idx_i64 = arith.index_cast %idx : index to i64
    %cmp = arith.cmpf ogt, %in, %out_val : f32
    %new_val = arith.select %cmp, %in, %out_val : f32
    %new_idx = arith.select %cmp, %idx_i64, %out_idx : i64
    linalg.yield %new_val, %new_idx : f32, i64
  } -> (tensor<2048xf32>, tensor<2048xi64>)
  iree_tensor_ext.dispatch.tensor.store %7#1, %1, offsets = [0], sizes = [2048], strides = [1] : tensor<2048xi64> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2048xi64>>
  return
}
