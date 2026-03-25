// Softmax along d1 (inner) with transposed output: tensor<1024x2048xf16> -> tensor<2048x1024xf16>
// Standard softmax computation but output is stored transposed.
// Input contiguous dim = d1, output contiguous dim = d0. Coalescing conflict on output.
//
// Source pattern: Attention patterns where softmax output feeds into a transposed matmul.
// Combines the softmax_medium and elementwise_transpose challenges.
//
// == Optimal Configuration ==
// Block size: 256 threads
// threads_per_row: 256 (all on d1 reduction)
// Vector width: 8 (f16 reads coalesced along d1)
// Workgroups: 1024 (one per row)
//
// Strategy: Compute softmax normally (all 4 stages). The final divide stage
// produces the output which needs to be written transposed.
//
// Two options for the transposed write:
// 1. Accept non-coalesced output writes (stride 1024 between consecutive d1 elements).
//    This is simpler but wastes output bandwidth.
// 2. Buffer the output row in shared memory (2048 * sizeof(f16) = 4KB per WG),
//    then re-read with transposed thread mapping for coalesced writes.
//    This costs 4KB smem but makes writes coalesced.
//
// For 2048 elements * f16, shared memory = 4KB, well within limits.
// The reduction stages already need shared memory for cross-warp reduce (~64 bytes).
// So option 2 adds minimal overhead.

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @softmax_transpose_output() {
  %c0 = arith.constant 0 : index
  %cst_neg_inf = arith.constant 0xFF800000 : f16
  %cst_zero = arith.constant 0.000000e+00 : f16
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024x2048xf16>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2048x1024xf16>>
  %in = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [1024, 2048], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024x2048xf16>> -> tensor<1024x2048xf16>

  // Stage 1: max reduction along d1
  %empty_row = tensor.empty() : tensor<1024xf16>
  %fill_neg_inf = linalg.fill ins(%cst_neg_inf : f16) outs(%empty_row : tensor<1024xf16>) -> tensor<1024xf16>
  %max_val = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]}
    ins(%in : tensor<1024x2048xf16>) outs(%fill_neg_inf : tensor<1024xf16>) {
  ^bb0(%a: f16, %b: f16):
    %v = arith.maximumf %a, %b : f16
    linalg.yield %v : f16
  } -> tensor<1024xf16>

  // Stage 2: subtract max + exp
  %empty_full = tensor.empty() : tensor<1024x2048xf16>
  %exp_val = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%in, %max_val : tensor<1024x2048xf16>, tensor<1024xf16>)
    outs(%empty_full : tensor<1024x2048xf16>) {
  ^bb0(%a: f16, %m: f16, %out: f16):
    %sub = arith.subf %a, %m : f16
    %e = math.exp %sub : f16
    linalg.yield %e : f16
  } -> tensor<1024x2048xf16>

  // Stage 3: sum reduction along d1
  %fill_zero = linalg.fill ins(%cst_zero : f16) outs(%empty_row : tensor<1024xf16>) -> tensor<1024xf16>
  %sum_val = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]}
    ins(%exp_val : tensor<1024x2048xf16>) outs(%fill_zero : tensor<1024xf16>) {
  ^bb0(%a: f16, %b: f16):
    %v = arith.addf %a, %b : f16
    linalg.yield %v : f16
  } -> tensor<1024xf16>

  // Stage 4: divide + TRANSPOSE OUTPUT
  // Output indexing: (d0, d1) -> (d1, d0) -- TRANSPOSED
  %empty_out = tensor.empty() : tensor<2048x1024xf16>
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>,
                     affine_map<(d0, d1) -> (d1, d0)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%exp_val, %sum_val : tensor<1024x2048xf16>, tensor<1024xf16>)
    outs(%empty_out : tensor<2048x1024xf16>) {
  ^bb0(%e: f16, %s: f16, %out: f16):
    %v = arith.divf %e, %s : f16
    linalg.yield %v : f16
  } -> tensor<2048x1024xf16>

  iree_tensor_ext.dispatch.tensor.store %result, %1, offsets = [0, 0], sizes = [2048, 1024], strides = [1, 1] : tensor<2048x1024xf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2048x1024xf16>>
  return
}
