// outer_reduction_dynamic.mlir
// Sum along d0 (outer dim): tensor<?x?xf32> -> tensor<?xf32>
// Pattern: Dynamic outer reduction. Same structure as static outer_reduction
// but dimensions loaded from hal.interface.constant.load.
//
// == Optimal Configuration ==
// Block size: 2D block (32, 16) = 512 threads (same as static case).
//   threadIdx.x = 32: maps to output dimension d1 (coalesced).
//   threadIdx.y = 16: cooperates on reduction dimension d0.
// Grid: (ceildiv(N, 32), 1) blocks where N = dynamic d1 size.
// Vector width: 1 or 4 depending on runtime alignment of N.
//   If N % 4 == 0, vec=4 for output writes. Otherwise scalar fallback.
// Elements per thread (reduction): ceildiv(M, 16) rows per threadIdx.y thread,
//   where M = dynamic d0 size.
// Reduction strategy: Same as static — smem tree reduction along threadIdx.y.
// Coalescing analysis: Same as static — threadIdx.x maps to contiguous d1.
//   Masking required: threads with tx >= N or rows >= M must be masked out.
// Thread mapping:
//   threadIdx.x -> d1 (output spatial dimension)
//   threadIdx.y -> d0 (reduction dimension)
// Dynamic handling: hal.interface.constant.load provides M and N.
//   No special-casing — same algorithm with bounds checks.

#pipeline_layout = #hal.pipeline.layout<constants = 2, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @outer_reduction_dynamic() {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %dim_M = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %dim_N = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf32>>{%dim_M, %dim_N}
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xf32>>{%dim_N}
  %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [%dim_M, %dim_N], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf32>>{%dim_M, %dim_N} -> tensor<?x?xf32>
  %3 = tensor.empty(%dim_N) : tensor<?xf32>
  %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<?xf32>) -> tensor<?xf32>
  %5 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d1)>],
    iterator_types = ["reduction", "parallel"]}
    ins(%2 : tensor<?x?xf32>) outs(%4 : tensor<?xf32>) {
  ^bb0(%in: f32, %out: f32):
    %6 = arith.addf %in, %out : f32
    linalg.yield %6 : f32
  } -> tensor<?xf32>
  iree_tensor_ext.dispatch.tensor.store %5, %1, offsets = [0], sizes = [%dim_N], strides = [1] : tensor<?xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xf32>>{%dim_N}
  return
}
