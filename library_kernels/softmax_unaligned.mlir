// Softmax on tensor<1024x511xf16>, reduce along d1.
// Pattern: Non-power-of-2 feature dimension. Needs masking or padding.
// 511 is prime — cannot be evenly divided by any useful thread count.
//
// == Optimal Configuration ==
// Block size: 256 threads (8 warps). Next power-of-2 >= 511 is 512, but
//   512 threads with 1 element each wastes half a warp. 256 threads with
//   2 elements each is more efficient.
// Threads per row: 256 (one block per row)
// Vector width: 1 (511 is not aligned to any vector width > 1).
//   Using vec2 would require 256 threads x 2 = 512 > 511, so the last
//   thread's second element is out-of-bounds and must be masked.
//   Scalar loads avoid this complexity.
// Elements per thread: ceil(511 / 256) = 2 (255 threads handle 2, 1 thread
//   handles 1). Thread k handles indices [2k] and [2k+1] if 2k+1 < 511.
//   Alternative: 512 threads, 1 element each, mask threads 511..511 (1 idle).
// Data fits in registers: YES — 2 x f16 = 4 bytes/thread. Trivially fits.
// Multi-pass: NO — single pass.
// Reduction strategy: Two-level (warp shuffle + cross-warp smem).
//   Masked threads contribute identity (-inf for max, 0 for sum).
//   Thread 255 only processes index 510 (511th element), index 511 is OOB.
// Number of workgroups: 1024 (one per row)
// Coalescing: Threads read consecutive f16 along d1 (innermost).
//   With 2 elements/thread using stride-2 access pattern: thread k reads
//   indices 2k and 2k+1. Alternatively, thread k reads k and k+256.
//   The k/k+256 pattern gives 2 coalesced loads of 256 consecutive elements.
//   The 2k/2k+1 pattern gives 256 coalesced loads of 2 consecutive elements.
//   Prefer k/k+256 for coalescing: first load covers [0..255], second [256..510].
//   Second load: only 255 valid elements, thread 255 is masked for idx 511.
// Softmax multi-stage:
//   Stage 1 (max): Load 2 elements (mask last), local max, shuffle + smem → max.
//   Stage 2 (sub+exp): Subtract max, exp. Masked element stays 0.
//   Stage 3 (sum): Sum of exp values (masked = 0), shuffle + smem → sum.
//   Stage 4 (div): Divide by sum, store (skip masked position).
//   Total: 1 load + 1 store per valid element. Optimal.

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @softmax_unaligned() {
  %cst_min = arith.constant 0xFC00 : f16
  %cst_zero = arith.constant 0.000000e+00 : f16
  %c0 = arith.constant 0 : index

  %input_binding = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024x511xf16>>
  %output_binding = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1024x511xf16>>

  %input = iree_tensor_ext.dispatch.tensor.load %input_binding, offsets = [0, 0], sizes = [1024, 511], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024x511xf16>> -> tensor<1024x511xf16>

  // Op1: Max reduction along d1.
  %empty_max = tensor.empty() : tensor<1024xf16>
  %fill_max = linalg.fill ins(%cst_min : f16) outs(%empty_max : tensor<1024xf16>) -> tensor<1024xf16>
  %max = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]}
    ins(%input : tensor<1024x511xf16>) outs(%fill_max : tensor<1024xf16>) {
  ^bb0(%in: f16, %out: f16):
    %0 = arith.maximumf %in, %out : f16
    linalg.yield %0 : f16
  } -> tensor<1024xf16>

  // Op2: Subtract max (broadcast) + exp.
  %empty_exp = tensor.empty() : tensor<1024x511xf16>
  %exp = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%input, %max : tensor<1024x511xf16>, tensor<1024xf16>) outs(%empty_exp : tensor<1024x511xf16>) {
  ^bb0(%in: f16, %mx: f16, %out: f16):
    %0 = arith.subf %in, %mx : f16
    %1 = math.exp %0 : f16
    linalg.yield %1 : f16
  } -> tensor<1024x511xf16>

  // Op3: Sum reduction along d1.
  %empty_sum = tensor.empty() : tensor<1024xf16>
  %fill_sum = linalg.fill ins(%cst_zero : f16) outs(%empty_sum : tensor<1024xf16>) -> tensor<1024xf16>
  %sum = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]}
    ins(%exp : tensor<1024x511xf16>) outs(%fill_sum : tensor<1024xf16>) {
  ^bb0(%in: f16, %out: f16):
    %0 = arith.addf %in, %out : f16
    linalg.yield %0 : f16
  } -> tensor<1024xf16>

  // Op4: Divide by sum (broadcast).
  %empty_out = tensor.empty() : tensor<1024x511xf16>
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%exp, %sum : tensor<1024x511xf16>, tensor<1024xf16>) outs(%empty_out : tensor<1024x511xf16>) {
  ^bb0(%in: f16, %s: f16, %out: f16):
    %0 = arith.divf %in, %s : f16
    linalg.yield %0 : f16
  } -> tensor<1024x511xf16>

  iree_tensor_ext.dispatch.tensor.store %result, %output_binding, offsets = [0, 0], sizes = [1024, 511], strides = [1, 1] : tensor<1024x511xf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1024x511xf16>>
  return
}
