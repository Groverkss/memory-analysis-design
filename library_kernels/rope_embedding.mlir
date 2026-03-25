// RoPE (Rotary Position Embedding): tensor<4x2048x32x128xf16>
// Pattern: Unsloth/Liger-Kernel. Split head_dim (128) in half.
// For each position, apply rotation using precomputed sin/cos tables:
//   out[..., :64]  = x[..., :64] * cos - x[..., 64:] * sin
//   out[..., 64:]  = x[..., 64:] * cos + x[..., :64] * sin
// The sin/cos tables are tensor<2048x64xf16> (seq_len x half_head_dim),
// broadcast across batch and heads.
//
// == Optimal Configuration ==
// Block size: 128 threads
// Vector width: 8 (128 bits / 16 bits = 8 f16 elements)
// Elements per thread: 8 (processing 8 elements of the half_head_dim at a time)
// Thread-to-data mapping: Map threads to the innermost dim (head_dim = 128).
//   128 threads cover the full head_dim. Each thread handles one element from
//   the first half and its corresponding element in the second half.
//   Actually: 128 threads / 2 halves = 64 threads per half, each processing
//   1 element. OR: use 64 threads with vec operations across the 64 half-dim.
//   Simplest: 128 threads, thread t processes element t. If t < 64, it computes
//   out[t] = x[t]*cos[t] - x[t+64]*sin[t]. If t >= 64, out[t] = x[t]*cos[t-64] + x[t-64]*sin[t-64].
// Grid shape: 4 * 2048 * 32 = 262144 workgroups (one per [batch, seq, head]).
//   Each workgroup handles one 128-element head_dim vector.
// Coalescing: head_dim (128) is the innermost dim. 128 threads each read one
//   f16 element = 256 bytes per transaction. However, each thread reads TWO
//   elements (x[t] and x[t +/- 64]), so the second read is strided by 64*2=128 bytes.
//   Both reads are within the same 128-element vector, both coalesced within a warp.
//   Sin/cos: tensor<2048x64xf16>, broadcast over batch and heads.
//   2048*64*2 = 256 KB, fits in L2 cache after first batch/head access.
// Memory traffic: Input read: 4*2048*32*128*2 = 64 MB.
//   Sin/cos read: 2*2048*64*2 = 512 KB (from L2 after first access).
//   Output write: 64 MB. Total ~128 MB.
// Compute: 4 muls + 1 add/sub per element = 5 FLOPs/elem.
//   Arithmetic intensity = 5 / 4 = 1.25 FLOP/byte. Memory-bound.

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @rope_embedding() {
  %c0 = arith.constant 0 : index

  // Input: [batch=4, seq=2048, heads=32, head_dim=128]
  %input_binding = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x2048x32x128xf16>>
  // Cos table: [seq=2048, half_head=64], broadcast over batch and heads
  %cos_binding = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048x64xf16>>
  // Sin table: [seq=2048, half_head=64], broadcast over batch and heads
  %sin_binding = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048x64xf16>>
  %output_binding = hal.interface.binding.subspan layout(#pipeline_layout) binding(3) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x2048x32x128xf16>>

  %input = iree_tensor_ext.dispatch.tensor.load %input_binding, offsets = [0, 0, 0, 0], sizes = [4, 2048, 32, 128], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x2048x32x128xf16>> -> tensor<4x2048x32x128xf16>
  %cos = iree_tensor_ext.dispatch.tensor.load %cos_binding, offsets = [0, 0], sizes = [2048, 64], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048x64xf16>> -> tensor<2048x64xf16>
  %sin = iree_tensor_ext.dispatch.tensor.load %sin_binding, offsets = [0, 0], sizes = [2048, 64], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048x64xf16>> -> tensor<2048x64xf16>

  // Extract first half [0:64] and second half [64:128] of head_dim.
  %first_half = tensor.extract_slice %input[0, 0, 0, 0] [4, 2048, 32, 64] [1, 1, 1, 1] : tensor<4x2048x32x128xf16> to tensor<4x2048x32x64xf16>
  %second_half = tensor.extract_slice %input[0, 0, 0, 64] [4, 2048, 32, 64] [1, 1, 1, 1] : tensor<4x2048x32x128xf16> to tensor<4x2048x32x64xf16>

  // Compute rotated first half: x1 * cos - x2 * sin
  %empty_first = tensor.empty() : tensor<4x2048x32x64xf16>
  %rotated_first = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d1, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d1, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    ins(%first_half, %second_half, %cos, %sin : tensor<4x2048x32x64xf16>, tensor<4x2048x32x64xf16>, tensor<2048x64xf16>, tensor<2048x64xf16>) outs(%empty_first : tensor<4x2048x32x64xf16>) {
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
    ins(%second_half, %first_half, %cos, %sin : tensor<4x2048x32x64xf16>, tensor<4x2048x32x64xf16>, tensor<2048x64xf16>, tensor<2048x64xf16>) outs(%empty_second : tensor<4x2048x32x64xf16>) {
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

  iree_tensor_ext.dispatch.tensor.store %output, %output_binding, offsets = [0, 0, 0, 0], sizes = [4, 2048, 32, 128], strides = [1, 1, 1, 1] : tensor<4x2048x32x128xf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x2048x32x128xf16>>
  return
}
