// matvec_f16.mlir
// Matrix-vector multiply: tensor<4096x4096xf16> * tensor<4096xf16> -> tensor<4096xf16>
// y = A * x, where A is [4096, 4096] and x is [4096].
// Pattern: llama.cpp GEMV. One block per output row. Inner product along columns.
// This is an inner reduction: for each output row i, y[i] = sum_j A[i,j] * x[j].
//
// == Optimal Configuration ==
// Block size: 128 threads (1D).
// Grid: 4096 blocks (one per output row).
// Vector width: 8 (f16 vec8 = 16 bytes, fills 128-bit load for both A and x).
// Elements per thread: 4096 / 128 = 32 columns per thread.
//   With vec=8: 32 / 8 = 4 vectorized iterations per thread.
//   CUB MemBoundScaling: items_per_thread = 32 * 4 / sizeof(f16) = 64.
// Thread-to-column mapping: thread k handles columns [k*32, k*32+32).
//   Or strided: thread k handles columns k, k+128, k+256, ... (32 elements).
//   Strided is better for coalescing: threads 0..127 read A[row, 0..127] together.
// Accumulator strategy: Each thread accumulates 32 multiply-adds in f16 (or f32
//   for accuracy). Then warp shuffle reduction across 32 threads in a warp.
//   128 threads = 4 warps -> 4 partial sums reduced via smem or final warp.
// Coalescing analysis:
//   Matrix A: row-major [4096, 4096]. Threads read consecutive columns in the same
//     row -> perfectly coalesced. 128 threads * f16 = 256 bytes per load.
//   Vector x: broadcast-like access. All threads in all blocks read the same x[j],
//     but at different times. x fits in L2 cache (4096 * 2 = 8KB).
// Reduction strategy: Warp shuffle for inner product reduction.
// Note: 128 threads (not 256) because GEMV is bandwidth-bound and fewer threads
//   means less register pressure for the accumulator. Each thread does more work.

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @matvec_f16() {
  %cst = arith.constant 0.000000e+00 : f16
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x4096xf16>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096xf16>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4096xf16>>
  %mat = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [4096, 4096], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x4096xf16>> -> tensor<4096x4096xf16>
  %vec = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0], sizes = [4096], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096xf16>> -> tensor<4096xf16>
  %3 = tensor.empty() : tensor<4096xf16>
  %4 = linalg.fill ins(%cst : f16) outs(%3 : tensor<4096xf16>) -> tensor<4096xf16>
  %5 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d1)>,
                     affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]}
    ins(%mat, %vec : tensor<4096x4096xf16>, tensor<4096xf16>)
    outs(%4 : tensor<4096xf16>) {
  ^bb0(%a: f16, %x: f16, %out: f16):
    %mul = arith.mulf %a, %x : f16
    %add = arith.addf %mul, %out : f16
    linalg.yield %add : f16
  } -> tensor<4096xf16>
  iree_tensor_ext.dispatch.tensor.store %5, %2, offsets = [0], sizes = [4096], strides = [1] : tensor<4096xf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4096xf16>>
  return
}
