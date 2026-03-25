// SiLU activation: silu(x) = x * sigmoid(x)
// Pattern: Unsloth/FlashInfer fused SiLU. Single input, single output elementwise.
// Compute: Cast f16->f32, negate, exp, add 1, div (sigmoid), multiply by x, cast back.
//
// == Optimal Configuration ==
// Block size: 128 threads
// Vector width: 8 (128 bits / 16 bits per f16 = 8 elements per vector load/store)
// Elements per thread: 8
// Elements per block: 128 * 8 = 1024
// Grid shape: (2048 * 4096) / 1024 = 8192 workgroups (flat 1D)
// Thread-to-data mapping: Flat 1D. Linearize [2048, 4096] -> [8388608].
//   Thread t in block b processes elements [b*1024 + t*8, b*1024 + t*8 + 7].
// Coalescing: Consecutive threads read consecutive f16 elements along the
//   innermost dim (d1, size 4096). 128 threads * 2 bytes = 256 bytes per
//   transaction, perfectly coalesced. 4096 / (128*8) = 4 iterations per row.
// Memory traffic: 2 * 2048 * 4096 * 2 bytes = 32 MB (read + write f16).
// Compute: sigmoid requires ~6 FLOPs (neg, exp, add, div) + 1 mul = 7 FLOPs/elem.
//   Arithmetic intensity = 7 / 4 = 1.75 FLOP/byte. Memory-bound on all GPUs.

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @silu_elementwise() {
  %c0 = arith.constant 0 : index
  %cst_one = arith.constant 1.000000e+00 : f32

  %input_binding = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048x4096xf16>>
  %output_binding = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2048x4096xf16>>

  %input = iree_tensor_ext.dispatch.tensor.load %input_binding, offsets = [0, 0], sizes = [2048, 4096], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048x4096xf16>> -> tensor<2048x4096xf16>

  %empty = tensor.empty() : tensor<2048x4096xf16>
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%input : tensor<2048x4096xf16>) outs(%empty : tensor<2048x4096xf16>) {
  ^bb0(%in: f16, %out: f16):
    // silu(x) = x * sigmoid(x) = x * (1 / (1 + exp(-x)))
    %x_f32 = arith.extf %in : f16 to f32
    %neg = arith.negf %x_f32 : f32
    %exp = math.exp %neg : f32
    %denom = arith.addf %exp, %cst_one : f32
    %sigmoid = arith.divf %cst_one, %denom : f32
    %silu = arith.mulf %x_f32, %sigmoid : f32
    %result = arith.truncf %silu : f32 to f16
    linalg.yield %result : f16
  } -> tensor<2048x4096xf16>

  iree_tensor_ext.dispatch.tensor.store %result, %output_binding, offsets = [0, 0], sizes = [2048, 4096], strides = [1, 1] : tensor<2048x4096xf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2048x4096xf16>>
  return
}
