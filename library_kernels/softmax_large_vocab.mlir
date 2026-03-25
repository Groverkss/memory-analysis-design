// Softmax on tensor<4096x32000xf16>, reduce along d1.
// Pattern: Cross-entropy/vocab softmax from Liger-Kernel. Needs chunked
// reduction because 32000 elements exceed register capacity.
//
// == Optimal Configuration ==
// Block size: 1024 threads (32 warps per block)
// Threads per row: 1024 (one block per row)
// Vector width: 8 (f16 vec8 = 16 bytes = one 128-bit load)
// Elements per thread: 32000 / 1024 ≈ 31.25 → need multi-pass loop
// Data fits in registers: NO — 31+ f16 values per thread is feasible for
//   registers alone (~62 bytes), but the non-uniform distribution (31 or 32
//   elements per thread due to 32000 not dividing 1024 evenly) means a loop
//   is cleaner. Liger-Kernel uses BLOCK_SIZE=32768 chunking.
// Multi-pass: YES — loop over chunks of the reduction dimension.
//   With 1024 threads x vec8, each pass covers 8192 elements.
//   ceil(32000/8192) = 4 passes for max, then 4 passes for sub+exp+sum,
//   then 4 passes for div+store. Total: 3 loads + 1 store per element
//   (2 extra loads vs fused approach, but necessary for register pressure).
//   Alternative: 2-pass "online softmax" (max+sum fused, then div) = 2 loads
//   + 1 store — but IREE models this as separate ops.
// Reduction strategy: Two-level per pass.
//   Intra-warp: butterfly shuffle (5 rounds).
//   Inter-warp: shared memory (32 partial results reduced by one warp).
//   Between passes: accumulate in registers.
// Number of workgroups: 4096 (one per row)
// Coalescing: Consecutive threads read consecutive f16 along d1 (innermost).
//   1024 threads x 2 bytes = 2048 bytes = 16 x 128-byte cache lines.
//   With vec8: 32 threads/warp x 16 bytes = 512 bytes = 4 cache lines.
//   Perfectly coalesced.
// Softmax multi-stage:
//   Pass 1 (max): Loop over 4 chunks, each thread reduces its vec8 chunk,
//     warp shuffle + smem reduce after all chunks → global row max.
//   Pass 2 (sub+exp+sum): Loop over 4 chunks, reload data, subtract max,
//     exp, accumulate local sum. After loop: warp shuffle + smem → row sum.
//     Intermediate exp values stored to output buffer (avoids 3rd load).
//   Pass 3 (div): Loop over 4 chunks, load exp values from output, divide
//     by sum, store final result.
//   Memory traffic: 2 loads input + 1 write exp + 1 load exp + 1 write final
//   = 2R + 1W + 1R + 1W = 3R + 2W per element.

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @softmax_large_vocab() {
  %cst_min = arith.constant 0xFC00 : f16
  %cst_zero = arith.constant 0.000000e+00 : f16
  %c0 = arith.constant 0 : index

  %input_binding = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x32000xf16>>
  %output_binding = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4096x32000xf16>>

  %input = iree_tensor_ext.dispatch.tensor.load %input_binding, offsets = [0, 0], sizes = [4096, 32000], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x32000xf16>> -> tensor<4096x32000xf16>

  // Op1: Max reduction along d1.
  %empty_max = tensor.empty() : tensor<4096xf16>
  %fill_max = linalg.fill ins(%cst_min : f16) outs(%empty_max : tensor<4096xf16>) -> tensor<4096xf16>
  %max = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]}
    ins(%input : tensor<4096x32000xf16>) outs(%fill_max : tensor<4096xf16>) {
  ^bb0(%in: f16, %out: f16):
    %0 = arith.maximumf %in, %out : f16
    linalg.yield %0 : f16
  } -> tensor<4096xf16>

  // Op2: Subtract max (broadcast) + exp.
  %empty_exp = tensor.empty() : tensor<4096x32000xf16>
  %exp = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%input, %max : tensor<4096x32000xf16>, tensor<4096xf16>) outs(%empty_exp : tensor<4096x32000xf16>) {
  ^bb0(%in: f16, %mx: f16, %out: f16):
    %0 = arith.subf %in, %mx : f16
    %1 = math.exp %0 : f16
    linalg.yield %1 : f16
  } -> tensor<4096x32000xf16>

  // Op3: Sum reduction along d1.
  %empty_sum = tensor.empty() : tensor<4096xf16>
  %fill_sum = linalg.fill ins(%cst_zero : f16) outs(%empty_sum : tensor<4096xf16>) -> tensor<4096xf16>
  %sum = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]}
    ins(%exp : tensor<4096x32000xf16>) outs(%fill_sum : tensor<4096xf16>) {
  ^bb0(%in: f16, %out: f16):
    %0 = arith.addf %in, %out : f16
    linalg.yield %0 : f16
  } -> tensor<4096xf16>

  // Op4: Divide by sum (broadcast).
  %empty_out = tensor.empty() : tensor<4096x32000xf16>
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%exp, %sum : tensor<4096x32000xf16>, tensor<4096xf16>) outs(%empty_out : tensor<4096x32000xf16>) {
  ^bb0(%in: f16, %s: f16, %out: f16):
    %0 = arith.divf %in, %s : f16
    linalg.yield %0 : f16
  } -> tensor<4096x32000xf16>

  iree_tensor_ext.dispatch.tensor.store %result, %output_binding, offsets = [0, 0], sizes = [4096, 32000], strides = [1, 1] : tensor<4096x32000xf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4096x32000xf16>>
  return
}
