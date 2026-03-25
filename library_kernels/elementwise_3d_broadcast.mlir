// Residual + bias: tensor<8x512x4096xf16> + tensor<4096xf16> -> tensor<8x512x4096xf16>
// Pattern: Residual connection + bias in transformer. Weight (bias/layernorm weight)
// is 1D (hidden_dim=4096), broadcast across batch (8) and sequence (512) dims.
//
// == Optimal Configuration ==
// Block size: 128 threads
// Vector width: 8 (128 bits / 16 bits = 8 f16 elements)
// Elements per thread: 8
// Elements per block: 128 * 8 = 1024
// Total elements: 8 * 512 * 4096 = 16777216 (16M)
// Grid shape: 16M / 1024 = 16384 workgroups (flat 1D)
// Thread-to-data mapping: Flat 1D over linearized [8, 512, 4096].
//   Thread t in block b: elements [b*1024 + t*8, b*1024 + t*8 + 7].
//   Bias index = linear_idx % 4096.
// Coalescing analysis:
//   Input (8x512x4096): d2 (4096) is innermost. Consecutive threads read
//   consecutive d2 elements. 128 threads * 2 bytes = 256 bytes. Coalesced.
//   Output: same layout, coalesced.
//   Bias (4096): broadcast operand. 4096 * 2 = 8 KB, trivially fits in L1/L2.
//   Accessed with same stride as d2, so consecutive threads read consecutive
//   bias elements within a warp. After first of 8*512=4096 rows, fully cached.
// Broadcast operand handling:
//   Bias is only 8 KB. After the first row's compulsory miss, all subsequent
//   rows (4095 of them) hit L2 cache. The effective bandwidth cost of the
//   bias is 8 KB / (8*512*4096*2 bytes) = 0.05% overhead. Negligible.
//   The compiler can also hoist the bias load to registers if tile size along
//   d2 matches the bias size, but this is unlikely for 4096 elements.
// Memory traffic: Read 16M*2 + 8KB. Write 16M*2. Total ~64 MB.

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @elementwise_3d_broadcast() {
  %c0 = arith.constant 0 : index

  %input_binding = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x512x4096xf16>>
  %bias_binding = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096xf16>>
  %output_binding = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8x512x4096xf16>>

  %input = iree_tensor_ext.dispatch.tensor.load %input_binding, offsets = [0, 0, 0], sizes = [8, 512, 4096], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x512x4096xf16>> -> tensor<8x512x4096xf16>
  %bias = iree_tensor_ext.dispatch.tensor.load %bias_binding, offsets = [0], sizes = [4096], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096xf16>> -> tensor<4096xf16>

  %empty = tensor.empty() : tensor<8x512x4096xf16>
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                     affine_map<(d0, d1, d2) -> (d2)>,
                     affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
    iterator_types = ["parallel", "parallel", "parallel"]}
    ins(%input, %bias : tensor<8x512x4096xf16>, tensor<4096xf16>) outs(%empty : tensor<8x512x4096xf16>) {
  ^bb0(%in: f16, %b: f16, %out: f16):
    %in_f32 = arith.extf %in : f16 to f32
    %b_f32 = arith.extf %b : f16 to f32
    %sum = arith.addf %in_f32, %b_f32 : f32
    %result = arith.truncf %sum : f32 to f16
    linalg.yield %result : f16
  } -> tensor<8x512x4096xf16>

  iree_tensor_ext.dispatch.tensor.store %result, %output_binding, offsets = [0, 0, 0], sizes = [8, 512, 4096], strides = [1, 1, 1] : tensor<8x512x4096xf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8x512x4096xf16>>
  return
}
