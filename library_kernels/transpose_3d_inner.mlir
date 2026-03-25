// 3D transpose swapping last two dims: tensor<64x512x1024xf32> -> tensor<64x1024x512xf32>
// Common in attention (transposing K for K^T in QK^T).
//
// Source pattern: ThunderKittens MHA (explicit K transpose via shared tile),
// PyTorch spatial softmax, CUTLASS epilogue threadmap.
//
// == Optimal Configuration ==
// Block size: 256 threads (32x8 2D block for the inner 2 dims)
// Tile: 32x32 elements of the inner 2 dims in shared memory
// Shared memory: 32 * 33 * sizeof(f32) = 4224 bytes (padding for bank conflicts)
// Grid: 64 * (1024/32) * (512/32) = 64 * 32 * 16 = 32768 workgroups
// Vector width: 4 (f32, 128-bit loads)
// d0 (batch=64) is purely parallel, tiled across grid.
// d1 and d2 are both parallel but transposed between input and output.
// Strategy: Same shared-memory transpose as 2D, but batched over d0.
// Each workgroup handles one 32x32 slice of (d1, d2) for one batch element.

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @transpose_3d_inner() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x512x1024xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<64x1024x512xf32>>
  %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [64, 512, 1024], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x512x1024xf32>> -> tensor<64x512x1024xf32>
  %3 = tensor.empty() : tensor<64x1024x512xf32>
  %4 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                     affine_map<(d0, d1, d2) -> (d0, d2, d1)>],
    iterator_types = ["parallel", "parallel", "parallel"]}
    ins(%2 : tensor<64x512x1024xf32>) outs(%3 : tensor<64x1024x512xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<64x1024x512xf32>
  iree_tensor_ext.dispatch.tensor.store %4, %1, offsets = [0, 0, 0], sizes = [64, 1024, 512], strides = [1, 1, 1] : tensor<64x1024x512xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<64x1024x512xf32>>
  return
}
