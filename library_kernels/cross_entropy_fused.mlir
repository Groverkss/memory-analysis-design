// cross_entropy_fused.mlir
// Fused logsumexp: tensor<2048x32000xf16> -> tensor<2048xf16>
// Pattern: Liger-Kernel/Unsloth cross-entropy fused kernel.
// Computes logsumexp(x) = log(sum(exp(x - max(x)))) + max(x) per row.
// Multi-op dispatch: max reduction, subtract+exp elementwise, sum reduction, log.
//
// This is a multi-op fusion expressed as separate linalg ops in the same dispatch.
// The compiler should fuse these into a single kernel that reads input once (or twice
// for the two-pass pattern) and writes one output per row.
//
// == Optimal Configuration ==
// Block size: 512 threads (1D).
// Grid: 2048 blocks (one per row).
// Vector width: 8 (f16 vec8 = 16 bytes).
// Elements per thread: 32000 / 512 = 62.5 -> ~63 elements/thread.
//   With vec=8: ~8 vectorized iterations per thread.
//   CUB MemBoundScaling: items_per_thread = 63 * 4 / sizeof(f16) = 126.
// Reduction strategy: Two reductions fused into one kernel.
//   Option A (two-pass, simple): Pass 1 computes max. Pass 2 computes sum(exp(x - max)).
//     Each pass is a warp shuffle reduction. Input read twice (2 * 2048 * 32000 * 2 = 256MB).
//   Option B (online/one-pass, advanced): Welford-like online logsumexp.
//     Carry (max, sum_exp) pair. On new element x:
//       if x > max: sum_exp = sum_exp * exp(max - x) + 1; max = x
//       else: sum_exp = sum_exp + exp(x - max)
//     Single pass over data. More compute per element but half the memory traffic.
//   IREE: The multi-op dispatch below represents Option A. The compiler can fuse.
// Coalescing: thread k reads [row, k], [row, k+512], ... -> perfectly coalesced.
// Note: 32000 vocab size is common in LLMs (LLaMA, GPT-2). Not power-of-2, needs masking.
//   The fused kernel avoids materializing the intermediate tensor<2048x32000xf16> for exp.

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @cross_entropy_fused() {
  %cst_neg_inf = arith.constant 0xFC00 : f16
  %cst_zero = arith.constant 0.000000e+00 : f16
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048x32000xf16>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2048xf16>>
  %input = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [2048, 32000], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048x32000xf16>> -> tensor<2048x32000xf16>

  // Step 1: Max reduction along d1.
  %empty_max = tensor.empty() : tensor<2048xf16>
  %fill_max = linalg.fill ins(%cst_neg_inf : f16) outs(%empty_max : tensor<2048xf16>) -> tensor<2048xf16>
  %row_max = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]}
    ins(%input : tensor<2048x32000xf16>) outs(%fill_max : tensor<2048xf16>) {
  ^bb0(%in: f16, %out: f16):
    %v = arith.maximumf %in, %out : f16
    linalg.yield %v : f16
  } -> tensor<2048xf16>

  // Step 2: Subtract max and exponentiate (elementwise, fused).
  %empty_exp = tensor.empty() : tensor<2048x32000xf16>
  %exp_shifted = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%input, %row_max : tensor<2048x32000xf16>, tensor<2048xf16>)
    outs(%empty_exp : tensor<2048x32000xf16>) {
  ^bb0(%x: f16, %mx: f16, %out: f16):
    %shifted = arith.subf %x, %mx : f16
    %exp_val = math.exp %shifted : f16
    linalg.yield %exp_val : f16
  } -> tensor<2048x32000xf16>

  // Step 3: Sum reduction of exp values along d1.
  %empty_sum = tensor.empty() : tensor<2048xf16>
  %fill_sum = linalg.fill ins(%cst_zero : f16) outs(%empty_sum : tensor<2048xf16>) -> tensor<2048xf16>
  %row_sum_exp = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]}
    ins(%exp_shifted : tensor<2048x32000xf16>) outs(%fill_sum : tensor<2048xf16>) {
  ^bb0(%in: f16, %out: f16):
    %v = arith.addf %in, %out : f16
    linalg.yield %v : f16
  } -> tensor<2048xf16>

  // Step 4: log(sum_exp) + max = logsumexp (elementwise).
  %empty_result = tensor.empty() : tensor<2048xf16>
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>,
                     affine_map<(d0) -> (d0)>,
                     affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]}
    ins(%row_sum_exp, %row_max : tensor<2048xf16>, tensor<2048xf16>)
    outs(%empty_result : tensor<2048xf16>) {
  ^bb0(%sum_val: f16, %max_val: f16, %out: f16):
    %log_val = math.log %sum_val : f16
    %lse = arith.addf %log_val, %max_val : f16
    linalg.yield %lse : f16
  } -> tensor<2048xf16>

  iree_tensor_ext.dispatch.tensor.store %result, %1, offsets = [0], sizes = [2048], strides = [1] : tensor<2048xf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2048xf16>>
  return
}
