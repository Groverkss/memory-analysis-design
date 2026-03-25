// Softmax on tensor<?x?xf32>, reduce along d1.
// Pattern: Dynamic sequence length attention. Both batch and feature dims
// are unknown at compile time.
//
// == Optimal Configuration ==
// Block size: Must be chosen conservatively or via runtime dispatch.
//   Typical: 256 threads. If d1 is small (< 128), 128 threads suffices.
//   If d1 is large (> 4096), 1024 threads is better.
//   Without knowing d1, 256 is a safe default.
// Threads per row: 256 (one block per row)
// Vector width: 1 (cannot vectorize safely without knowing alignment/size).
//   If d1 is known to be a multiple of 4 at runtime, vec4 is possible,
//   but the compiler must emit scalar fallback or masked vector loads.
// Elements per thread: d1 / 256 (runtime value, looped)
// Data fits in registers: UNKNOWN — depends on d1. For d1 <= 1024,
//   4 elements/thread fits. For d1 = 32000, 125 elements/thread needs loop.
// Multi-pass: YES (must assume large d1 to be safe)
// Reduction strategy: Two-level (warp shuffle + cross-warp smem).
//   Thread count for reduction is fixed at 256 regardless of d1.
//   Loop bounds are ceil(d1 / 256). Out-of-bounds threads contribute
//   identity element (-inf for max, 0 for sum).
// Number of workgroups: d0 (one per row, unknown at compile time)
// Coalescing: If d1 is innermost (row-major), consecutive threads read
//   consecutive elements → coalesced. The dynamic size doesn't affect
//   coalescing, only the tail handling (last incomplete cache line).
// Softmax multi-stage:
//   Same as static cases but with dynamic loop bounds:
//   Stage 1 (max): for i in range(0, d1, 256): load, max. Guard: i+tid < d1.
//   Stage 2 (sub+exp): same loop, sub max, exp, store to output.
//   Stage 3 (sum): same loop over output, accumulate sum.
//   Stage 4 (div): same loop, load exp, divide, store.
//   Alternatively, stages 2+3 can be fused if exp values fit in registers.

#pipeline_layout = #hal.pipeline.layout<constants = 2, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @softmax_dynamic() {
  %cst_min = arith.constant 0xFF800000 : f32
  %cst_zero = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index

  // Load dynamic dimensions.
  %dim0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %dim1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index

  %input_binding = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf32>>{%dim0, %dim1}
  %output_binding = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?xf32>>{%dim0, %dim1}

  %input = iree_tensor_ext.dispatch.tensor.load %input_binding, offsets = [0, 0], sizes = [%dim0, %dim1], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf32>>{%dim0, %dim1} -> tensor<?x?xf32>

  // Op1: Max reduction along d1.
  %empty_max = tensor.empty(%dim0) : tensor<?xf32>
  %fill_max = linalg.fill ins(%cst_min : f32) outs(%empty_max : tensor<?xf32>) -> tensor<?xf32>
  %max = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]}
    ins(%input : tensor<?x?xf32>) outs(%fill_max : tensor<?xf32>) {
  ^bb0(%in: f32, %out: f32):
    %0 = arith.maximumf %in, %out : f32
    linalg.yield %0 : f32
  } -> tensor<?xf32>

  // Op2: Subtract max (broadcast) + exp.
  %empty_exp = tensor.empty(%dim0, %dim1) : tensor<?x?xf32>
  %exp = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%input, %max : tensor<?x?xf32>, tensor<?xf32>) outs(%empty_exp : tensor<?x?xf32>) {
  ^bb0(%in: f32, %mx: f32, %out: f32):
    %0 = arith.subf %in, %mx : f32
    %1 = math.exp %0 : f32
    linalg.yield %1 : f32
  } -> tensor<?x?xf32>

  // Op3: Sum reduction along d1.
  %empty_sum = tensor.empty(%dim0) : tensor<?xf32>
  %fill_sum = linalg.fill ins(%cst_zero : f32) outs(%empty_sum : tensor<?xf32>) -> tensor<?xf32>
  %sum = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]}
    ins(%exp : tensor<?x?xf32>) outs(%fill_sum : tensor<?xf32>) {
  ^bb0(%in: f32, %out: f32):
    %0 = arith.addf %in, %out : f32
    linalg.yield %0 : f32
  } -> tensor<?xf32>

  // Op4: Divide by sum (broadcast).
  %empty_out = tensor.empty(%dim0, %dim1) : tensor<?x?xf32>
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%exp, %sum : tensor<?x?xf32>, tensor<?xf32>) outs(%empty_out : tensor<?x?xf32>) {
  ^bb0(%in: f32, %s: f32, %out: f32):
    %0 = arith.divf %in, %s : f32
    linalg.yield %0 : f32
  } -> tensor<?x?xf32>

  iree_tensor_ext.dispatch.tensor.store %result, %output_binding, offsets = [0, 0], sizes = [%dim0, %dim1], strides = [1, 1] : tensor<?x?xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?xf32>>{%dim0, %dim1}
  return
}
