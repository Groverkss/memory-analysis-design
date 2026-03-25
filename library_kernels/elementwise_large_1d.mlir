// Scale: tensor<16777216xf16> -> tensor<16777216xf16>
// Pattern: Large flat tensor (16M = 2^24 elements). Pure 1D grid-stride loop.
// Multiply every element by a constant scale factor. Simplest possible kernel.
//
// == Optimal Configuration ==
// Block size: 128 threads
// Vector width: 8 (128 bits / 16 bits = 8 f16 elements per vector load/store)
// Elements per thread: 8
// Elements per block: 128 * 8 = 1024
// Grid shape: 16777216 / 1024 = 16384 workgroups (1D)
// Thread-to-data mapping: Pure flat 1D.
//   Block b, thread t processes elements [b*1024 + t*8, b*1024 + t*8 + 7].
//   No grid-stride loop needed — each thread processes exactly 8 elements
//   and there are enough blocks to cover all 16M elements.
// Coalescing: Trivially perfect. 1D contiguous tensor, consecutive threads
//   read consecutive elements. 128 threads * 2 bytes = 256 bytes per
//   transaction, filling 2 full 128-byte cache lines.
// Memory traffic: 16M * 2 bytes * 2 (read + write) = 64 MB.
// Compute: 1 mul per element. Arithmetic intensity = 1 / 4 = 0.25 FLOP/byte.
//   Extremely memory-bound.
// Occupancy: 16384 blocks across ~100 SMs = ~164 blocks/SM queued.
//   With 128 threads/block, can fit 16 blocks/SM (2048 threads/SM) = full occupancy.
// Alignment: 16M * 2 bytes = 32 MB, 64-byte aligned. All vector loads aligned.

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @elementwise_large_1d() {
  %c0 = arith.constant 0 : index
  %cst_scale = arith.constant 0.0078125 : f16

  %input_binding = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16777216xf16>>
  %output_binding = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16777216xf16>>

  %input = iree_tensor_ext.dispatch.tensor.load %input_binding, offsets = [0], sizes = [16777216], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16777216xf16>> -> tensor<16777216xf16>

  %empty = tensor.empty() : tensor<16777216xf16>
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>,
                     affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]}
    ins(%input : tensor<16777216xf16>) outs(%empty : tensor<16777216xf16>) {
  ^bb0(%in: f16, %out: f16):
    %scaled = arith.mulf %in, %cst_scale : f16
    linalg.yield %scaled : f16
  } -> tensor<16777216xf16>

  iree_tensor_ext.dispatch.tensor.store %result, %output_binding, offsets = [0], sizes = [16777216], strides = [1] : tensor<16777216xf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16777216xf16>>
  return
}
