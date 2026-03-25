// GEGLU: gelu_tanh(gate) * up
// Pattern: Liger-Kernel. Uses tanh approximation of GELU for the gate path.
// gelu_tanh(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// Heavier compute than SwiGLU due to tanh + polynomial.
//
// == Optimal Configuration ==
// Block size: 128 threads
// Vector width: 8 (128 bits / 16 bits = 8 f16 elements)
// Elements per thread: 8
// Elements per block: 128 * 8 = 1024
// Grid shape: (2048 * 4096) / 1024 = 8192 workgroups (flat 1D)
// Thread-to-data mapping: Flat 1D over linearized [2048, 4096].
// Coalescing: Same as SwiGLU — d1 innermost for all three tensors.
//   128 threads * 2 bytes = 256 bytes, perfectly coalesced.
// Memory traffic: 3 * 2048 * 4096 * 2 bytes = 48 MB (2 reads + 1 write).
// Compute: GELU-tanh ~15 FLOPs (x^3, scale, tanh polynomial, add, mul) + mul(up).
//   Arithmetic intensity = 16 / 6 = 2.67 FLOP/byte. Still memory-bound on
//   modern GPUs (need ~50+ FLOP/byte to be compute-bound).
// Note: tanh is the expensive part. Hardware tanh (SFU on NVIDIA) takes ~8 cycles,
//   but pipeline hides this at memory-bound occupancy levels.

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @geglu_fused() {
  %c0 = arith.constant 0 : index
  %cst_half = arith.constant 0.500000e+00 : f32
  %cst_one = arith.constant 1.000000e+00 : f32
  %cst_coeff = arith.constant 0.044715 : f32
  // sqrt(2/pi) = 0.7978845608...
  %cst_sqrt2pi = arith.constant 0.7978845608 : f32

  %gate_binding = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048x4096xf16>>
  %up_binding = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048x4096xf16>>
  %output_binding = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2048x4096xf16>>

  %gate = iree_tensor_ext.dispatch.tensor.load %gate_binding, offsets = [0, 0], sizes = [2048, 4096], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048x4096xf16>> -> tensor<2048x4096xf16>
  %up = iree_tensor_ext.dispatch.tensor.load %up_binding, offsets = [0, 0], sizes = [2048, 4096], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048x4096xf16>> -> tensor<2048x4096xf16>

  %empty = tensor.empty() : tensor<2048x4096xf16>
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%gate, %up : tensor<2048x4096xf16>, tensor<2048x4096xf16>) outs(%empty : tensor<2048x4096xf16>) {
  ^bb0(%gate_val: f16, %up_val: f16, %out: f16):
    // geglu(gate, up) = gelu_tanh(gate) * up
    // gelu_tanh(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    %g = arith.extf %gate_val : f16 to f32
    %u = arith.extf %up_val : f16 to f32
    %g2 = arith.mulf %g, %g : f32
    %g3 = arith.mulf %g2, %g : f32
    %coeff_g3 = arith.mulf %cst_coeff, %g3 : f32
    %inner = arith.addf %g, %coeff_g3 : f32
    %scaled = arith.mulf %cst_sqrt2pi, %inner : f32
    %tanh_val = math.tanh %scaled : f32
    %one_plus_tanh = arith.addf %cst_one, %tanh_val : f32
    %half_x = arith.mulf %cst_half, %g : f32
    %gelu = arith.mulf %half_x, %one_plus_tanh : f32
    %geglu = arith.mulf %gelu, %u : f32
    %result = arith.truncf %geglu : f32 to f16
    linalg.yield %result : f16
  } -> tensor<2048x4096xf16>

  iree_tensor_ext.dispatch.tensor.store %result, %output_binding, offsets = [0, 0], sizes = [2048, 4096], strides = [1, 1] : tensor<2048x4096xf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2048x4096xf16>>
  return
}
