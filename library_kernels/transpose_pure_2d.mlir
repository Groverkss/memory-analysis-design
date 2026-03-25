// Pure 2D transpose: tensor<2048x4096xf16> -> tensor<4096x2048xf16>
// No computation, just data movement with transposed layout.
// This is the classic shared-memory transpose problem.
//
// Source pattern: TVM transpose rule (16x8 thread tile, smem with +1 padding),
// CUTLASS PitchLinearStripminedThreadMap, ThunderKittens shared tile swizzle.
//
// == Optimal Configuration ==
// Block size: 128 threads (16x8 2D block)
// Tile: 32x32 elements in shared memory (2KB for f16)
// Shared memory: 32 * 33 * sizeof(f16) = 2112 bytes (+1 col padding for bank conflicts)
// Grid: (4096/32) x (2048/32) = 128 x 64 = 8192 workgroups
// Vector width: 8 (f16, 128-bit loads)
// Strategy:
//   1. Each block loads a 32x32 tile from input with coalesced reads (threads along d1)
//   2. Store to shared memory in row-major with padding (33-wide rows)
//   3. __syncthreads
//   4. Read from shared memory transposed (threads along d0 of the original)
//   5. Write to output coalesced (threads along d1 of output = d0 of input)
// Both input reads AND output writes are coalesced. Shared memory absorbs the transpose.
// This is the ONLY correct solution for transpose -- v1's approach of dropping
// coalescing is wrong; shared memory promotion is needed.

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @transpose_pure_2d() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048x4096xf16>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4096x2048xf16>>
  %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [2048, 4096], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048x4096xf16>> -> tensor<2048x4096xf16>
  %3 = tensor.empty() : tensor<4096x2048xf16>
  %4 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d1, d0)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%2 : tensor<2048x4096xf16>) outs(%3 : tensor<4096x2048xf16>) {
  ^bb0(%in: f16, %out: f16):
    linalg.yield %in : f16
  } -> tensor<4096x2048xf16>
  iree_tensor_ext.dispatch.tensor.store %4, %1, offsets = [0, 0], sizes = [4096, 2048], strides = [1, 1] : tensor<4096x2048xf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4096x2048xf16>>
  return
}
