// SiLU on single-row wide tensor: tensor<1x65536xf16> -> tensor<1x65536xf16>
// Pattern: Single batch element but very wide hidden dimension.
// Common in inference with batch_size=1 (e.g., autoregressive decoding with
// large intermediate FFN activations: 4 * hidden_dim = 4 * 16384 = 65536).
//
// == Optimal Configuration ==
// Block size: 512 threads (or 1024 for maximum utilization of the single row)
// Vector width: 8 (128 bits / 16 bits = 8 f16 elements)
// Elements per thread: 8
// Elements per block: 512 * 8 = 4096
// Grid shape: 65536 / 4096 = 16 workgroups (1D along d1)
//   With 1024 threads: 1024 * 8 = 8192 per block, 65536 / 8192 = 8 workgroups.
// Thread-to-data mapping: Flat 1D along d1 (the only meaningful dimension).
//   d0 = 1 contributes no parallelism. All threads distributed along d1.
//   Thread t in block b: elements [b*4096 + t*8, b*4096 + t*8 + 7].
// Coalescing: d1 (65536) is innermost. Perfectly coalesced.
//   512 threads * 2 bytes = 1024 bytes per transaction.
// Occupancy concern: Only 16 workgroups (at 512 threads/block).
//   On a GPU with 100 SMs, only 16 SMs are active. 84% of SMs idle.
//   This is the fundamental problem with tiny-batch kernels.
//   Mitigation: Use 512+ threads/block to maximize ILP within each SM.
//   With 1024 threads/block and 8 blocks, it's even worse for occupancy but
//   each SM does more work, hiding memory latency better.
// Alternative: Treat as flat 1D tensor<65536xf16>. Same optimal config.
//   The batch dim of 1 is a no-op for scheduling.
// Memory traffic: 65536 * 2 * 2 = 256 KB. Tiny. Latency-dominated.

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @elementwise_tiny_batch() {
  %c0 = arith.constant 0 : index
  %cst_one = arith.constant 1.000000e+00 : f32

  %input_binding = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x65536xf16>>
  %output_binding = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x65536xf16>>

  %input = iree_tensor_ext.dispatch.tensor.load %input_binding, offsets = [0, 0], sizes = [1, 65536], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x65536xf16>> -> tensor<1x65536xf16>

  %empty = tensor.empty() : tensor<1x65536xf16>
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%input : tensor<1x65536xf16>) outs(%empty : tensor<1x65536xf16>) {
  ^bb0(%in: f16, %out: f16):
    // silu(x) = x * sigmoid(x)
    %x_f32 = arith.extf %in : f16 to f32
    %neg = arith.negf %x_f32 : f32
    %exp = math.exp %neg : f32
    %denom = arith.addf %exp, %cst_one : f32
    %sigmoid = arith.divf %cst_one, %denom : f32
    %silu = arith.mulf %x_f32, %sigmoid : f32
    %result = arith.truncf %silu : f32 to f16
    linalg.yield %result : f16
  } -> tensor<1x65536xf16>

  iree_tensor_ext.dispatch.tensor.store %result, %output_binding, offsets = [0, 0], sizes = [1, 65536], strides = [1, 1] : tensor<1x65536xf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x65536xf16>>
  return
}
