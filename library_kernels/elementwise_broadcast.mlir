// Add bias with broadcast: tensor<2048x4096xf32> + tensor<4096xf32> -> tensor<2048x4096xf32>
// Pattern: Bias addition in transformer FFN/attention output. Bias is 1D (hidden_dim),
// broadcast along the batch/sequence dim (d0).
//
// == Optimal Configuration ==
// Block size: 256 threads
// Vector width: 4 (128 bits / 32 bits per f32 = 4 elements per vector load/store)
// Elements per thread: 4
// Elements per block: 256 * 4 = 1024
// Grid shape: (2048 * 4096) / 1024 = 8192 workgroups (flat 1D)
// Thread-to-data mapping: Flat 1D over linearized [2048, 4096].
//   Thread t in block b processes output elements [b*1024 + t*4, b*1024 + t*4 + 3].
//   Bias index = (linear_idx % 4096), so threads in a warp accessing consecutive
//   elements along d1 read consecutive bias elements.
// Coalescing analysis:
//   Input (2048x4096): d1 is innermost, threads read consecutive d1 elements.
//   256 threads * 4 bytes = 1024 bytes per transaction. Perfectly coalesced.
//   Output (2048x4096): same layout, perfectly coalesced.
//   Bias (4096): threads in a warp read consecutive elements from the bias vector.
//   Since multiple rows share the same bias values, the bias vector is reused
//   across 2048 rows. After the first row's access, bias lives in L2 cache.
// Broadcast operand handling:
//   The bias vector is 4096 * 4 = 16 KB, fits entirely in L2 cache (and even L1
//   on most GPUs). First access is a compulsory miss; subsequent rows hit L2.
//   Effective extra bandwidth from bias is negligible: 16 KB vs 32 MB for main tensor.
// Memory traffic: Read 2048*4096*4 = 32 MB + bias 16 KB. Write 32 MB. Total ~64 MB.

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @elementwise_broadcast() {
  %c0 = arith.constant 0 : index

  %input_binding = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048x4096xf32>>
  %bias_binding = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096xf32>>
  %output_binding = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2048x4096xf32>>

  %input = iree_tensor_ext.dispatch.tensor.load %input_binding, offsets = [0, 0], sizes = [2048, 4096], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048x4096xf32>> -> tensor<2048x4096xf32>
  %bias = iree_tensor_ext.dispatch.tensor.load %bias_binding, offsets = [0], sizes = [4096], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096xf32>> -> tensor<4096xf32>

  %empty = tensor.empty() : tensor<2048x4096xf32>
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%input, %bias : tensor<2048x4096xf32>, tensor<4096xf32>) outs(%empty : tensor<2048x4096xf32>) {
  ^bb0(%in: f32, %b: f32, %out: f32):
    %sum = arith.addf %in, %b : f32
    linalg.yield %sum : f32
  } -> tensor<2048x4096xf32>

  iree_tensor_ext.dispatch.tensor.store %result, %output_binding, offsets = [0, 0], sizes = [2048, 4096], strides = [1, 1] : tensor<2048x4096xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2048x4096xf32>>
  return
}
