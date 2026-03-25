// SwiGLU: silu(gate) * up
// Pattern: Liger-Kernel/Unsloth. Two input tensors (gate, up), one output.
// This is the gating mechanism in LLaMA-style FFN blocks.
//
// == Optimal Configuration ==
// Block size: 128 threads
// Vector width: 8 (128 bits / 16 bits = 8 f16 elements per vector load/store)
// Elements per thread: 8
// Elements per block: 128 * 8 = 1024
// Grid shape: (2048 * 4096) / 1024 = 8192 workgroups (flat 1D)
// Thread-to-data mapping: Flat 1D over linearized [2048, 4096].
//   Thread t in block b: elements [b*1024 + t*8, b*1024 + t*8 + 7].
// Coalescing: Both inputs and output have d1 (4096) as innermost dim.
//   128 threads * 2 bytes = 256 bytes per transaction. Perfectly coalesced
//   for all three tensors. Coalescing target met for gate, up, and output.
// Memory traffic: 2 reads + 1 write = 3 * 2048 * 4096 * 2 bytes = 48 MB.
// Compute: sigmoid(gate) ~6 FLOPs + mul(gate, sigmoid) + mul(silu, up) = ~8 FLOPs/elem.
//   Arithmetic intensity = 8 / 6 = 1.33 FLOP/byte. Memory-bound.

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @swiglu_fused() {
  %c0 = arith.constant 0 : index
  %cst_one = arith.constant 1.000000e+00 : f32

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
    // swiglu(gate, up) = silu(gate) * up = gate * sigmoid(gate) * up
    %g = arith.extf %gate_val : f16 to f32
    %u = arith.extf %up_val : f16 to f32
    %neg = arith.negf %g : f32
    %exp = math.exp %neg : f32
    %denom = arith.addf %exp, %cst_one : f32
    %sigmoid = arith.divf %cst_one, %denom : f32
    %silu = arith.mulf %g, %sigmoid : f32
    %swiglu = arith.mulf %silu, %u : f32
    %result = arith.truncf %swiglu : f32 to f16
    linalg.yield %result : f16
  } -> tensor<2048x4096xf16>

  iree_tensor_ext.dispatch.tensor.store %result, %output_binding, offsets = [0, 0], sizes = [2048, 4096], strides = [1, 1] : tensor<2048x4096xf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2048x4096xf16>>
  return
}
