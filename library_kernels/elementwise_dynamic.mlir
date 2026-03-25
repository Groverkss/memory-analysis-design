// ReLU with dynamic shapes: tensor<?x?xf32> -> tensor<?x?xf32>
// Pattern: Dynamic shape elementwise. Shapes unknown at compile time.
// relu(x) = max(x, 0)
//
// == Optimal Configuration ==
// Block size: 256 threads
// Vector width: 4 (128 bits / 32 bits = 4 f32 elements, when alignment permits)
// Elements per thread: 4 (when innermost dim is aligned; scalar fallback otherwise)
// Grid shape: ceil(d0 * d1 / 1024) workgroups — computed at dispatch time.
// Thread-to-data mapping: Flat 1D over linearized [d0, d1].
//   This is the safest choice for dynamic shapes: avoids dimension-dependent
//   thread mapping that could under-utilize threads for small dims.
// Bounds checking:
//   - Total element count = d0 * d1 (loaded from interface constants).
//   - Each thread computes its linear index: block_id * 1024 + thread_id * 4.
//   - If linear_index + 3 < total_elements: process full vector (4 elements).
//   - If linear_index >= total_elements: early exit (no work).
//   - Otherwise: scalar loop for remaining 1-3 elements (tail handling).
//   The compiler emits this bounds check automatically from the dynamic tensor.
// Coalescing: Assuming row-major layout (innermost dim contiguous), consecutive
//   threads access consecutive elements in the innermost dim. Coalesced as long
//   as d1 >= warp_size (32). If d1 < 32, partial coalescing with wasted bandwidth.
// Vectorization caveat: If d1 is not divisible by 4, the compiler must use
//   scalar loads for the boundary columns. This is the main cost of dynamic shapes.
// Memory traffic: d0 * d1 * 4 bytes * 2 (read + write).

#pipeline_layout = #hal.pipeline.layout<constants = 2, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @elementwise_dynamic() {
  %c0 = arith.constant 0 : index
  %cst_zero = arith.constant 0.000000e+00 : f32

  %dim0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %dim1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index

  %input_binding = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf32>>{%dim0, %dim1}
  %output_binding = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?xf32>>{%dim0, %dim1}

  %input = iree_tensor_ext.dispatch.tensor.load %input_binding, offsets = [0, 0], sizes = [%dim0, %dim1], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf32>>{%dim0, %dim1} -> tensor<?x?xf32>

  %empty = tensor.empty(%dim0, %dim1) : tensor<?x?xf32>
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%input : tensor<?x?xf32>) outs(%empty : tensor<?x?xf32>) {
  ^bb0(%in: f32, %out: f32):
    %relu = arith.maximumf %in, %cst_zero : f32
    linalg.yield %relu : f32
  } -> tensor<?x?xf32>

  iree_tensor_ext.dispatch.tensor.store %result, %output_binding, offsets = [0, 0], sizes = [%dim0, %dim1], strides = [1, 1] : tensor<?x?xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?xf32>>{%dim0, %dim1}
  return
}
