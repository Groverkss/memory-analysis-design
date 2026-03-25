// Softmax on tensor<2048x128256xf16>, reduce along d1.
// Pattern: Gemma vocab (128256 tokens) from Unsloth chunked cross-entropy.
// Far too large for a single block — needs cooperative multi-block or
// chunked approach.
//
// == Optimal Configuration ==
// Block size: 1024 threads (32 warps per block)
// Threads per row: 1024 per block, multiple blocks per row
// Vector width: 8 (f16 vec8 = 16 bytes)
// Elements per thread per pass: 8192 / 1024 = 8 per chunk (one vec8 load)
// Data fits in registers: NO — 128256 elements per row = 250 KB per row in
//   f16. Even with 1024 threads, that's ~250 bytes/thread if held all at
//   once. Must stream through data in chunks.
// Multi-pass: YES — essential. With 1024 threads x vec8 = 8192 elements/pass,
//   ceil(128256 / 8192) = 16 passes per stage.
// Reduction strategy: Three-level.
//   Level 1: Intra-warp butterfly shuffle (5 rounds).
//   Level 2: Cross-warp shared memory within block (32 → 1).
//   Level 3: Cross-block reduction via global memory atomics or
//     cooperative kernel launch (gridSync). Unsloth uses chunked approach:
//     split vocab into chunks of ~32K, compute partial logsumexp per chunk,
//     then combine. For pure softmax, a 2-pass global approach works:
//     Pass A: each block computes partial max/sum for its chunk → global buf.
//     Pass B: reduce partials, then normalize.
//   Alternatively (simpler): single block per row with 16 internal passes.
//   At 1024 threads this is feasible — 16 passes x ~10 cycles/element ≈
//   160K cycles ≈ 0.1ms at 1.5 GHz, dominated by memory latency anyway.
// Number of workgroups: 2048 (one per row, multi-pass within block)
//   If multi-block: 2048 x ceil(128256/8192) = 2048 x 16 = 32768 blocks
//   (but synchronization cost may not be worth it).
// Coalescing: Consecutive threads read consecutive f16 along d1 (innermost).
//   Same as softmax_large_vocab — perfectly coalesced.
// Softmax multi-stage:
//   Single-block approach (simpler, likely sufficient):
//     Stage 1 (max): 16 loop iterations, each loads vec8, reduces locally.
//       After loop: warp shuffle + smem → row max.
//     Stage 2 (sub+exp+sum): 16 iterations, reload, sub max, exp, sum locally.
//       Write exp to output buffer. After loop: shuffle + smem → row sum.
//     Stage 3 (div): 16 iterations, load exp from output, divide by sum, store.
//     Memory traffic: 2R input + 1W exp + 1R exp + 1W final = 3R + 2W.
//     Total data moved per row: 128256 x 2 bytes x 5 = 1.25 MB/row.
//   Multi-block approach (for very high throughput):
//     Phase 1: Each of 16 blocks per row computes local max → global scratch.
//     Phase 2: One block reduces 16 partial maxes → global max.
//     Phase 3: Each block computes local exp-sum → global scratch.
//     Phase 4: One block reduces 16 partial sums → global sum.
//     Phase 5: Each block normalizes its chunk.
//     Requires grid sync or separate kernel launches between phases.

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @softmax_very_large_vocab() {
  %cst_min = arith.constant 0xFC00 : f16
  %cst_zero = arith.constant 0.000000e+00 : f16
  %c0 = arith.constant 0 : index

  %input_binding = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048x128256xf16>>
  %output_binding = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2048x128256xf16>>

  %input = iree_tensor_ext.dispatch.tensor.load %input_binding, offsets = [0, 0], sizes = [2048, 128256], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048x128256xf16>> -> tensor<2048x128256xf16>

  // Op1: Max reduction along d1.
  %empty_max = tensor.empty() : tensor<2048xf16>
  %fill_max = linalg.fill ins(%cst_min : f16) outs(%empty_max : tensor<2048xf16>) -> tensor<2048xf16>
  %max = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]}
    ins(%input : tensor<2048x128256xf16>) outs(%fill_max : tensor<2048xf16>) {
  ^bb0(%in: f16, %out: f16):
    %0 = arith.maximumf %in, %out : f16
    linalg.yield %0 : f16
  } -> tensor<2048xf16>

  // Op2: Subtract max (broadcast) + exp.
  %empty_exp = tensor.empty() : tensor<2048x128256xf16>
  %exp = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%input, %max : tensor<2048x128256xf16>, tensor<2048xf16>) outs(%empty_exp : tensor<2048x128256xf16>) {
  ^bb0(%in: f16, %mx: f16, %out: f16):
    %0 = arith.subf %in, %mx : f16
    %1 = math.exp %0 : f16
    linalg.yield %1 : f16
  } -> tensor<2048x128256xf16>

  // Op3: Sum reduction along d1.
  %empty_sum = tensor.empty() : tensor<2048xf16>
  %fill_sum = linalg.fill ins(%cst_zero : f16) outs(%empty_sum : tensor<2048xf16>) -> tensor<2048xf16>
  %sum = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]}
    ins(%exp : tensor<2048x128256xf16>) outs(%fill_sum : tensor<2048xf16>) {
  ^bb0(%in: f16, %out: f16):
    %0 = arith.addf %in, %out : f16
    linalg.yield %0 : f16
  } -> tensor<2048xf16>

  // Op4: Divide by sum (broadcast).
  %empty_out = tensor.empty() : tensor<2048x128256xf16>
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%exp, %sum : tensor<2048x128256xf16>, tensor<2048xf16>) outs(%empty_out : tensor<2048x128256xf16>) {
  ^bb0(%in: f16, %s: f16, %out: f16):
    %0 = arith.divf %in, %s : f16
    linalg.yield %0 : f16
  } -> tensor<2048x128256xf16>

  iree_tensor_ext.dispatch.tensor.store %result, %output_binding, offsets = [0, 0], sizes = [2048, 128256], strides = [1, 1] : tensor<2048x128256xf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2048x128256xf16>>
  return
}
