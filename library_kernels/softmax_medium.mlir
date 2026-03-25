// Softmax on tensor<2048x4096xf16>, reduce along d1.
// Pattern: Standard attention softmax. One row per block, 256 threads.
//
// == Optimal Configuration ==
// Block size: 256 threads (8 warps per block)
// Threads per row: 256 (one block per row)
// Vector width: 8 (f16 vec8 = 16 bytes = one 128-bit load)
// Elements per thread: 4096 / 256 = 16 elements, loaded as 2 x vec8
// Data fits in registers: YES — 16 x f16 = 32 bytes/thread, easily fits.
//   Each thread holds its 16 elements across all 4 softmax stages.
// Multi-pass: NO — single pass, all 4096 elements covered by 256 threads
// Reduction strategy: Two-level.
//   Level 1: Warp shuffle butterfly within each of 8 warps (5 rounds).
//   Level 2: Cross-warp reduction via shared memory (8 partial results,
//   one warp reduces them). ~48 bytes shared memory for partials.
// Number of workgroups: 2048 (one per row)
// Coalescing: Threads read consecutive f16 elements along d1 (innermost).
//   256 threads x 2 bytes = 512 bytes per transaction group. With vec8,
//   each thread issues one 16-byte load, 32 threads in a warp produce
//   512-byte access — perfectly coalesced in 4 x 128-byte sectors.
// Softmax multi-stage:
//   Stage 1 (max): Load 16 elements into registers, local max, warp shuffle,
//     cross-warp smem reduce → row max.
//   Stage 2 (sub+exp): Subtract max from registers, compute exp in-place.
//     No reload needed — data stays in registers.
//   Stage 3 (sum): Local sum of exp values, warp shuffle, cross-warp smem
//     reduce → row sum.
//   Stage 4 (div): Divide registers by sum, store result.
//   Total memory traffic: 1 load + 1 store per element (optimal).

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @softmax_medium() {
  %cst_min = arith.constant 0xFC00 : f16
  %cst_zero = arith.constant 0.000000e+00 : f16
  %c0 = arith.constant 0 : index

  %input_binding = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048x4096xf16>>
  %output_binding = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2048x4096xf16>>

  %input = iree_tensor_ext.dispatch.tensor.load %input_binding, offsets = [0, 0], sizes = [2048, 4096], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048x4096xf16>> -> tensor<2048x4096xf16>

  // Op1: Max reduction along d1.
  %empty_max = tensor.empty() : tensor<2048xf16>
  %fill_max = linalg.fill ins(%cst_min : f16) outs(%empty_max : tensor<2048xf16>) -> tensor<2048xf16>
  %max = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]}
    ins(%input : tensor<2048x4096xf16>) outs(%fill_max : tensor<2048xf16>) {
  ^bb0(%in: f16, %out: f16):
    %0 = arith.maximumf %in, %out : f16
    linalg.yield %0 : f16
  } -> tensor<2048xf16>

  // Op2: Subtract max (broadcast) + exp.
  %empty_exp = tensor.empty() : tensor<2048x4096xf16>
  %exp = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%input, %max : tensor<2048x4096xf16>, tensor<2048xf16>) outs(%empty_exp : tensor<2048x4096xf16>) {
  ^bb0(%in: f16, %mx: f16, %out: f16):
    %0 = arith.subf %in, %mx : f16
    %1 = math.exp %0 : f16
    linalg.yield %1 : f16
  } -> tensor<2048x4096xf16>

  // Op3: Sum reduction along d1.
  %empty_sum = tensor.empty() : tensor<2048xf16>
  %fill_sum = linalg.fill ins(%cst_zero : f16) outs(%empty_sum : tensor<2048xf16>) -> tensor<2048xf16>
  %sum = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]}
    ins(%exp : tensor<2048x4096xf16>) outs(%fill_sum : tensor<2048xf16>) {
  ^bb0(%in: f16, %out: f16):
    %0 = arith.addf %in, %out : f16
    linalg.yield %0 : f16
  } -> tensor<2048xf16>

  // Op4: Divide by sum (broadcast).
  %empty_out = tensor.empty() : tensor<2048x4096xf16>
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%exp, %sum : tensor<2048x4096xf16>, tensor<2048xf16>) outs(%empty_out : tensor<2048x4096xf16>) {
  ^bb0(%in: f16, %s: f16, %out: f16):
    %0 = arith.divf %in, %s : f16
    linalg.yield %0 : f16
  } -> tensor<2048x4096xf16>

  iree_tensor_ext.dispatch.tensor.store %result, %output_binding, offsets = [0, 0], sizes = [2048, 4096], strides = [1, 1] : tensor<2048x4096xf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2048x4096xf16>>
  return
}
