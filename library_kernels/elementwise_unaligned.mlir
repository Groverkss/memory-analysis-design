// Abs with unaligned innermost dim: tensor<1000x3xf32> -> tensor<1000x3xf32>
// Pattern: Very small innermost dim (3). Cannot vectorize along d1.
// Common in 3D point cloud data, RGB images, spatial coordinates.
//
// == Optimal Configuration ==
// Block size: 256 threads (or 128; doesn't matter much for this shape)
// Vector width: 1 (innermost dim is 3, not divisible by any useful vector width).
//   Cannot use vec_width=4 (f32, 128-bit) because 3 % 4 != 0.
//   Attempting to vectorize would require masking, which adds overhead for no gain.
// Elements per thread: 1 (scalar)
// Thread-to-data mapping: Two viable strategies:
//   (a) Flat 1D: Linearize to 3000 elements. 256 threads * 1 = 256 per block.
//       ceil(3000/256) = 12 blocks. Simple but coalescing is imperfect: threads
//       within a warp access consecutive elements, but rows are only 3 wide so
//       a warp spans ~10 rows. Still coalesced since memory is contiguous.
//   (b) 2D: 1 thread per element in d1 (3 threads active out of 32 in a warp).
//       91% of threads wasted. Terrible. Don't do this.
//   Strategy (a) is optimal.
// Grid shape: ceil(3000 / 256) = 12 workgroups (1D)
// Coalescing analysis:
//   Flat 1D: threads read consecutive f32 elements from contiguous memory.
//   32 threads * 4 bytes = 128 bytes = exactly one cache line. Perfectly coalesced.
//   The row boundary at every 3rd element doesn't matter because the memory is
//   contiguous (row-major, no padding).
// Memory traffic: 3000 * 4 * 2 = 24 KB. Trivially small.
// Occupancy concern: Only 12 blocks total. On a GPU with 100 SMs, 88 SMs are
//   idle. This kernel will be latency-dominated, not bandwidth-dominated.
//   In practice, this dispatch should be fused with something else.

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @elementwise_unaligned() {
  %c0 = arith.constant 0 : index

  %input_binding = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1000x3xf32>>
  %output_binding = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1000x3xf32>>

  %input = iree_tensor_ext.dispatch.tensor.load %input_binding, offsets = [0, 0], sizes = [1000, 3], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1000x3xf32>> -> tensor<1000x3xf32>

  %empty = tensor.empty() : tensor<1000x3xf32>
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%input : tensor<1000x3xf32>) outs(%empty : tensor<1000x3xf32>) {
  ^bb0(%in: f32, %out: f32):
    %abs = math.absf %in : f32
    linalg.yield %abs : f32
  } -> tensor<1000x3xf32>

  iree_tensor_ext.dispatch.tensor.store %result, %output_binding, offsets = [0, 0], sizes = [1000, 3], strides = [1, 1] : tensor<1000x3xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1000x3xf32>>
  return
}
