// matvec_transposed.mlir
// Transposed matvec: y = A^T * x
// tensor<4096x4096xf16> (read as transposed) * tensor<4096xf16> -> tensor<4096xf16>
// y[j] = sum_i A[i, j] * x[i]
// Pattern: Outer reduction GEMV. The weight matrix A is stored row-major [4096, 4096],
// but we reduce along d0 (the row/first dimension), producing one output per column j.
// This means the reduction walks along the non-contiguous (strided) dimension of A.
//
// == Optimal Configuration ==
// Block size: 2D block (32, 8) = 256 threads.
//   threadIdx.x = 32: maps to output columns j (contiguous in A's inner dim -> coalesced reads).
//   threadIdx.y = 8: cooperates on the reduction dimension i (rows of A).
// Grid: (4096 / 32, 1) = (128, 1) blocks. Each block produces 32 output elements.
// Vector width: 8 (f16 vec8 = 16 bytes). threadIdx.x threads read consecutive columns
//   within the same row, so vec8 along d1 is coalesced.
// Elements per thread (reduction): 4096 / 8 = 512 rows per threadIdx.y thread.
//   Each thread loops over 512 rows, accumulating partial sums.
// Thread mapping:
//   threadIdx.x -> d1 (output spatial dimension = columns of A). Stride-1 in memory.
//   threadIdx.y -> d0 (reduction dimension = rows of A). Each ty handles rows
//     [ty * 512, (ty+1) * 512).
// Reduction strategy: Shared memory tree reduction along threadIdx.y.
//   Each threadIdx.y lane has a partial dot product for its 32 output columns.
//   smem[ty][tx] stores the partial; log2(8) = 3 barriers to reduce.
//   threadIdx.y == 0 writes 32 final results.
// Coalescing analysis:
//   Matrix A: row-major [4096, 4096]. Thread (tx, ty) reads A[row, tx_offset + tx].
//     32 consecutive tx threads read 32 consecutive columns in the same row -> coalesced.
//   Vector x: x[row] is the same for all tx within a row -> broadcast.
//     x is 4096 * 2 = 8KB, fits in L2. Each ty reads different x elements.
//   Output y: threadIdx.x maps to consecutive j -> coalesced writes.
// Why 2D block: Unlike normal matvec (inner reduction), transposed matvec has
//   reduction along the strided dim. Using threadIdx.y for cooperative reduction
//   keeps threadIdx.x aligned with the contiguous dim for coalescing.

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @matvec_transposed() {
  %cst = arith.constant 0.000000e+00 : f16
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x4096xf16>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096xf16>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4096xf16>>
  %mat = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [4096, 4096], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x4096xf16>> -> tensor<4096x4096xf16>
  %vec = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0], sizes = [4096], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096xf16>> -> tensor<4096xf16>
  %3 = tensor.empty() : tensor<4096xf16>
  %4 = linalg.fill ins(%cst : f16) outs(%3 : tensor<4096xf16>) -> tensor<4096xf16>
  // y[j] = sum_i A[i, j] * x[i]
  // d0 = i (reduction), d1 = j (parallel)
  %5 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>,
                     affine_map<(d0, d1) -> (d1)>],
    iterator_types = ["reduction", "parallel"]}
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
