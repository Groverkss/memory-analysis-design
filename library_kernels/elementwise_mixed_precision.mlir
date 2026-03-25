// FP8 block-wise dequantization: tensor<4096x4096xf8E4M3FNUZ> * tensor<128x128xf32> -> tensor<4096x4096xf16>
// Pattern: FlashInfer/Quack MXFP8. Per-block scales where each 32x32 block of
// the weight matrix shares a single f32 scale factor.
// Block size for quantization: 32x32 (4096/32 = 128 blocks per dim).
// Each element: dequant(x[i,j]) = cast_f16(cast_f32(x[i,j]) * scale[i/32, j/32])
//
// == Optimal Configuration ==
// Block size: 256 threads
// Vector width: 16 (128 bits / 8 bits = 16 f8 elements per vector load)
// Elements per thread: 16 (for input); output is f16 so vec_width=8 for stores
// Thread-to-data mapping: 2D tiling aligned to 32x32 quantization blocks.
//   Each workgroup handles one or more 32x32 blocks.
//   With 256 threads and 32x32=1024 elements per block: 4 elements/thread.
//   Or: each WG handles a 32x(32*N) strip for better coalescing.
// Grid shape: (4096/32) * (4096/32) = 128 * 128 = 16384 workgroups
//   if each WG handles exactly one 32x32 block.
// Coalescing analysis:
//   Input (f8): d1 is innermost. Threads read consecutive f8 elements along d1.
//   256 threads * 1 byte = 256 bytes per transaction. Perfectly coalesced.
//   Output (f16): d1 is innermost. 256 threads * 2 bytes = 512 bytes. Coalesced.
//   Scale: tensor<128x128xf32>. Each 32x32 block shares ONE scale value.
//   All threads in a WG read the same scale -> broadcast from L1/registers.
//   Scale tensor = 128*128*4 = 64 KB, fits in L2. Accessed once per block.
// Compute: 1 cast (f8->f32) + 1 mul + 1 cast (f32->f16) = 3 ops/element.
//   Arithmetic intensity = 3 / 3 bytes (1 read + 2 write) = 1 FLOP/byte. Memory-bound.

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @elementwise_mixed_precision() {
  %c0 = arith.constant 0 : index

  %input_binding = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x4096xf8E4M3FNUZ>>
  %scale_binding = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x128xf32>>
  %output_binding = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4096x4096xf16>>

  %input = iree_tensor_ext.dispatch.tensor.load %input_binding, offsets = [0, 0], sizes = [4096, 4096], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x4096xf8E4M3FNUZ>> -> tensor<4096x4096xf8E4M3FNUZ>
  %scale = iree_tensor_ext.dispatch.tensor.load %scale_binding, offsets = [0, 0], sizes = [128, 128], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x128xf32>> -> tensor<128x128xf32>

  %empty = tensor.empty() : tensor<4096x4096xf16>
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0 floordiv 32, d1 floordiv 32)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%input, %scale : tensor<4096x4096xf8E4M3FNUZ>, tensor<128x128xf32>) outs(%empty : tensor<4096x4096xf16>) {
  ^bb0(%in: f8E4M3FNUZ, %s: f32, %out: f16):
    %val = arith.extf %in : f8E4M3FNUZ to f32
    %scaled = arith.mulf %val, %s : f32
    %result = arith.truncf %scaled : f32 to f16
    linalg.yield %result : f16
  } -> tensor<4096x4096xf16>

  iree_tensor_ext.dispatch.tensor.store %result, %output_binding, offsets = [0, 0], sizes = [4096, 4096], strides = [1, 1] : tensor<4096x4096xf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4096x4096xf16>>
  return
}
