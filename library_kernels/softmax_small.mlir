// Softmax on tensor<512x128xf32>, reduce along d1.
// Pattern: Apex warp-level softmax. 128 elements fit in one warp
// (WARP_SIZE=32, WARP_ITERATIONS=4 elements/thread).
//
// == Optimal Configuration ==
// Block size: 128 threads (4 warps per block)
// Threads per row: 32 (one warp per row)
// Rows per block: 4 (WARP_BATCH=4, one warp per row, 4 warps)
// Vector width: 4 (128 elements / 32 threads = 4 elements/thread)
// Elements per thread: 4
// Data fits in registers: YES — 4 x f32 = 16 bytes/thread, trivially fits
// Multi-pass: NO — single pass over 128 elements
// Reduction strategy: Pure warp shuffle (butterfly reduction across 32 lanes).
//   ZERO shared memory needed. Each warp independently reduces its row.
// Number of workgroups: 512 / 4 = 128 workgroups
// Coalescing: Consecutive threads read consecutive elements along d1 (innermost).
//   Thread k in warp reads elements [4k, 4k+3]. Perfectly coalesced 128-byte
//   transactions (32 threads x 4 bytes = 128 bytes).
// Softmax multi-stage:
//   All 4 stages (max, sub+exp, sum, div) operate on the same 4 registers per
//   thread. Each reduction (max, sum) is a single warp shuffle butterfly —
//   5 shuffle rounds for 32 lanes. No synchronization between stages needed
//   since everything is warp-local.

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @softmax_small() {
  %cst_min = arith.constant 0xFF800000 : f32
  %cst_zero = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index

  // Bind input and output tensors.
  %input_binding = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<512x128xf32>>
  %output_binding = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<512x128xf32>>

  %input = iree_tensor_ext.dispatch.tensor.load %input_binding, offsets = [0, 0], sizes = [512, 128], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<512x128xf32>> -> tensor<512x128xf32>

  // Op1: Max reduction along d1.
  %empty_max = tensor.empty() : tensor<512xf32>
  %fill_max = linalg.fill ins(%cst_min : f32) outs(%empty_max : tensor<512xf32>) -> tensor<512xf32>
  %max = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]}
    ins(%input : tensor<512x128xf32>) outs(%fill_max : tensor<512xf32>) {
  ^bb0(%in: f32, %out: f32):
    %0 = arith.maximumf %in, %out : f32
    linalg.yield %0 : f32
  } -> tensor<512xf32>

  // Op2: Subtract max (broadcast) + exp.
  %empty_exp = tensor.empty() : tensor<512x128xf32>
  %exp = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%input, %max : tensor<512x128xf32>, tensor<512xf32>) outs(%empty_exp : tensor<512x128xf32>) {
  ^bb0(%in: f32, %mx: f32, %out: f32):
    %0 = arith.subf %in, %mx : f32
    %1 = math.exp %0 : f32
    linalg.yield %1 : f32
  } -> tensor<512x128xf32>

  // Op3: Sum reduction along d1.
  %empty_sum = tensor.empty() : tensor<512xf32>
  %fill_sum = linalg.fill ins(%cst_zero : f32) outs(%empty_sum : tensor<512xf32>) -> tensor<512xf32>
  %sum = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]}
    ins(%exp : tensor<512x128xf32>) outs(%fill_sum : tensor<512xf32>) {
  ^bb0(%in: f32, %out: f32):
    %0 = arith.addf %in, %out : f32
    linalg.yield %0 : f32
  } -> tensor<512xf32>

  // Op4: Divide by sum (broadcast).
  %empty_out = tensor.empty() : tensor<512x128xf32>
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%exp, %sum : tensor<512x128xf32>, tensor<512xf32>) outs(%empty_out : tensor<512x128xf32>) {
  ^bb0(%in: f32, %s: f32, %out: f32):
    %0 = arith.divf %in, %s : f32
    linalg.yield %0 : f32
  } -> tensor<512x128xf32>

  iree_tensor_ext.dispatch.tensor.store %result, %output_binding, offsets = [0, 0], sizes = [512, 128], strides = [1, 1] : tensor<512x128xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<512x128xf32>>
  return
}
