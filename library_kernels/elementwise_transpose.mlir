// Cast + scale with transposed output: f16 [1024, 4096] -> f32 [4096, 1024]
// Pattern: Coalescing conflict. Input contiguous dim is d1 (4096), output
// contiguous dim is d1 (1024) which maps to d0 of input. Cannot simultaneously
// coalesce reads and writes with a naive thread mapping.
//
// == Optimal Configuration ==
// Block size: 256 threads (8 warps)
// Tile shape: 32x32 (32 rows x 32 cols of input = 32 cols x 32 rows of output)
// Thread mapping: 256 threads handle 32x32 = 1024 elements. 4 elements/thread.
//   For the read phase: threads are mapped row-major in input space (d0, d1).
//   For the write phase: threads are mapped row-major in output space (d1, d0).
// Shared memory promotion strategy:
//   - Load 32x32 tile from input with coalesced reads (threads along d1).
//   - Store to shared memory as 32x33 (33 = 32 + 1 padding to avoid bank conflicts).
//   - __syncthreads()
//   - Load from shared memory with transposed indexing.
//   - Write to output with coalesced writes (threads along output d1 = input d0).
// Vector width: Read: 8 (f16, 128-bit). Write: 4 (f32, 128-bit).
//   But within 32x32 tile, effective vec width may be limited to 4 for f16
//   since tile width is 32 and we have 256 threads (32*32/256 = 4 elements/thread).
// Grid shape: (1024/32) x (4096/32) = 32 x 128 = 4096 workgroups (2D grid).
// Coalescing analysis:
//   Input read: threads in a warp access consecutive d1 elements -> coalesced.
//   Output write: after transpose through shared memory, threads in a warp
//   access consecutive output d1 elements -> coalesced.
//   Without shared memory promotion: either reads OR writes are strided by 1024
//   or 4096 elements, wasting 75-97% of each cache line.
// Memory traffic: Read 1024*4096*2 = 8 MB. Write 4096*1024*4 = 16 MB. Total 24 MB.

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @elementwise_transpose() {
  %c0 = arith.constant 0 : index
  %cst_scale = arith.constant 0.0078125 : f32

  %input_binding = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024x4096xf16>>
  %output_binding = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4096x1024xf32>>

  %input = iree_tensor_ext.dispatch.tensor.load %input_binding, offsets = [0, 0], sizes = [1024, 4096], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024x4096xf16>> -> tensor<1024x4096xf16>

  %empty = tensor.empty() : tensor<4096x1024xf32>
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%input : tensor<1024x4096xf16>) outs(%empty : tensor<4096x1024xf32>) {
  ^bb0(%in: f16, %out: f32):
    %val = arith.extf %in : f16 to f32
    %scaled = arith.mulf %val, %cst_scale : f32
    linalg.yield %scaled : f32
  } -> tensor<4096x1024xf32>

  iree_tensor_ext.dispatch.tensor.store %result, %output_binding, offsets = [0, 0], sizes = [4096, 1024], strides = [1, 1] : tensor<4096x1024xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4096x1024xf32>>
  return
}
