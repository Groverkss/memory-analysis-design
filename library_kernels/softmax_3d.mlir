// Softmax on tensor<8x16x2048xf16>, reduce along d2.
// Pattern: Multi-head attention softmax. Batch=8, heads=16, seq_len=2048.
// Parallel over batch*heads = 128 independent softmax rows.
//
// == Optimal Configuration ==
// Block size: 256 threads (8 warps per block)
// Threads per row: 256 (one block per (batch, head) pair)
// Vector width: 8 (f16 vec8 = 16 bytes)
// Elements per thread: 2048 / 256 = 8 (exactly one vec8 load)
// Data fits in registers: YES — 8 x f16 = 16 bytes/thread. Trivially fits.
//   All 4 stages operate on the same 8 registers. No reload needed.
// Multi-pass: NO — single pass covers all 2048 elements.
// Reduction strategy: Two-level.
//   Intra-warp: butterfly shuffle (5 rounds).
//   Inter-warp: shared memory (8 partial results → one warp reduces).
// Number of workgroups: 8 x 16 = 128 (one per batch-head pair)
//   Note: only 128 workgroups — may underutilize GPU (e.g., A100 has 108 SMs).
//   Occupancy is OK since 128 > 108, but just barely. Each SM gets ~1 block.
//   With 256 threads and minimal shared memory, could fit 4+ blocks/SM for
//   higher occupancy if we had more workgroups.
// Coalescing: Consecutive threads read consecutive f16 along d2 (innermost).
//   256 threads x 2 bytes = 512 bytes per transaction group. Perfectly
//   coalesced. d0 and d1 are batch dims — different workgroups, no concern.
// Softmax multi-stage:
//   Stage 1 (max): Each thread loads 8 elements (vec8), computes local max.
//     Warp shuffle → warp max. Smem → block max. Broadcast back.
//   Stage 2 (sub+exp): Subtract max from 8 registers, compute exp in-place.
//   Stage 3 (sum): Local sum of 8 exp values. Warp shuffle → smem → block sum.
//   Stage 4 (div): Divide 8 registers by sum. Store.
//   Total memory traffic: 1 load + 1 store per element. Optimal — fully fused
//   in registers with zero intermediate spilling.

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @softmax_3d() {
  %cst_min = arith.constant 0xFC00 : f16
  %cst_zero = arith.constant 0.000000e+00 : f16
  %c0 = arith.constant 0 : index

  %input_binding = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x16x2048xf16>>
  %output_binding = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8x16x2048xf16>>

  %input = iree_tensor_ext.dispatch.tensor.load %input_binding, offsets = [0, 0, 0], sizes = [8, 16, 2048], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x16x2048xf16>> -> tensor<8x16x2048xf16>

  // Op1: Max reduction along d2.
  %empty_max = tensor.empty() : tensor<8x16xf16>
  %fill_max = linalg.fill ins(%cst_min : f16) outs(%empty_max : tensor<8x16xf16>) -> tensor<8x16xf16>
  %max = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                     affine_map<(d0, d1, d2) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel", "reduction"]}
    ins(%input : tensor<8x16x2048xf16>) outs(%fill_max : tensor<8x16xf16>) {
  ^bb0(%in: f16, %out: f16):
    %0 = arith.maximumf %in, %out : f16
    linalg.yield %0 : f16
  } -> tensor<8x16xf16>

  // Op2: Subtract max (broadcast along d2) + exp.
  %empty_exp = tensor.empty() : tensor<8x16x2048xf16>
  %exp = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                     affine_map<(d0, d1, d2) -> (d0, d1)>,
                     affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
    iterator_types = ["parallel", "parallel", "parallel"]}
    ins(%input, %max : tensor<8x16x2048xf16>, tensor<8x16xf16>) outs(%empty_exp : tensor<8x16x2048xf16>) {
  ^bb0(%in: f16, %mx: f16, %out: f16):
    %0 = arith.subf %in, %mx : f16
    %1 = math.exp %0 : f16
    linalg.yield %1 : f16
  } -> tensor<8x16x2048xf16>

  // Op3: Sum reduction along d2.
  %empty_sum = tensor.empty() : tensor<8x16xf16>
  %fill_sum = linalg.fill ins(%cst_zero : f16) outs(%empty_sum : tensor<8x16xf16>) -> tensor<8x16xf16>
  %sum = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                     affine_map<(d0, d1, d2) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel", "reduction"]}
    ins(%exp : tensor<8x16x2048xf16>) outs(%fill_sum : tensor<8x16xf16>) {
  ^bb0(%in: f16, %out: f16):
    %0 = arith.addf %in, %out : f16
    linalg.yield %0 : f16
  } -> tensor<8x16xf16>

  // Op4: Divide by sum (broadcast along d2).
  %empty_out = tensor.empty() : tensor<8x16x2048xf16>
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                     affine_map<(d0, d1, d2) -> (d0, d1)>,
                     affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
    iterator_types = ["parallel", "parallel", "parallel"]}
    ins(%exp, %sum : tensor<8x16x2048xf16>, tensor<8x16xf16>) outs(%empty_out : tensor<8x16x2048xf16>) {
  ^bb0(%in: f16, %s: f16, %out: f16):
    %0 = arith.divf %in, %s : f16
    linalg.yield %0 : f16
  } -> tensor<8x16x2048xf16>

  iree_tensor_ext.dispatch.tensor.store %result, %output_binding, offsets = [0, 0, 0], sizes = [8, 16, 2048], strides = [1, 1, 1] : tensor<8x16x2048xf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8x16x2048xf16>>
  return
}
