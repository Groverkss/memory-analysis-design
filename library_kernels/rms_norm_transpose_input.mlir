// RMSNorm where the input is transposed: the hidden dim is NOT the innermost dim.
// Input: tensor<4096x2048xf16> where d0=hidden, d1=batch.
// We want to normalize each of the 2048 "rows", but a "row" is d0 (non-contiguous).
// Output: tensor<4096x2048xf16> (same layout).
//
// This is equivalent to normalizing columns of a row-major matrix.
// The reduction dimension (d0=4096) is NOT contiguous in memory.
//
// Source pattern: This is what happens with transposed activations, e.g., after
// a transpose that hasn't been materialized. llama.cpp handles this by refusing
// non-contiguous inputs. PyTorch dispatches to a strided kernel.
//
// == Optimal Configuration ==
// Block size: 512 threads (2D: block.x=32 parallel, block.y=16 reduction)
// Grid: ceil(2048/32) = 64 workgroups (tiling d1, the parallel/contiguous dim)
// Vector width: Cannot vectorize the reduction dim (it's strided).
//   But CAN vectorize the parallel dim (d1 is contiguous): output vec=8.
//
// Thread mapping:
//   threadIdx.x (32): maps to d1 (parallel, contiguous). Coalesced reads.
//   threadIdx.y (16): cooperates on d0 (reduction, NON-contiguous).
//   Each thread reads d0 positions with stride block.y, so reads at
//   d0 = threadIdx.y, threadIdx.y + 16, ..., up to 4096. Stride = 16 * 2048 elements.
//
// Memory access: Within each row of d0, reads are coalesced along d1 (good).
// But successive d0 steps jump by 2048 elements (stride). This is an outer
// reduction pattern.
//
// Reduction: threadIdx.y cooperates via shared memory tree reduction.
// Shared memory: 32 * 16 * sizeof(f32) = 2048 bytes for accumulators.
// Each threadIdx.x position needs its own reduction lane.

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @rms_norm_transpose_input() {
  %cst = arith.constant 0.000000e+00 : f32
  %cst_eps = arith.constant 1.0e-06 : f32
  %cst_n = arith.constant 4096.0 : f32
  %c0 = arith.constant 0 : index
  %input = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x2048xf16>>
  %weight = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096xf16>>
  %result = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4096x2048xf16>>
  %in = iree_tensor_ext.dispatch.tensor.load %input, offsets = [0, 0], sizes = [4096, 2048], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x2048xf16>> -> tensor<4096x2048xf16>
  %w = iree_tensor_ext.dispatch.tensor.load %weight, offsets = [0], sizes = [4096], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096xf16>> -> tensor<4096xf16>

  // Reduction along d0 (NON-contiguous) for each d1 position
  %empty_rstd = tensor.empty() : tensor<2048xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty_rstd : tensor<2048xf32>) -> tensor<2048xf32>

  // Op 1: Sum of squares along d0
  %sumsq = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d1)>],
    iterator_types = ["reduction", "parallel"]}
    ins(%in : tensor<4096x2048xf16>) outs(%fill : tensor<2048xf32>) {
  ^bb0(%x: f16, %acc: f32):
    %xf = arith.extf %x : f16 to f32
    %sq = arith.mulf %xf, %xf : f32
    %add = arith.addf %sq, %acc : f32
    linalg.yield %add : f32
  } -> tensor<2048xf32>

  // Op 2: rsqrt(mean_sq + eps)
  %rstd = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>,
                     affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]}
    ins(%sumsq : tensor<2048xf32>) outs(%empty_rstd : tensor<2048xf32>) {
  ^bb0(%s: f32, %out: f32):
    %mean = arith.divf %s, %cst_n : f32
    %eps = arith.addf %mean, %cst_eps : f32
    %rs = math.rsqrt %eps : f32
    linalg.yield %rs : f32
  } -> tensor<2048xf32>

  // Op 3: Normalize: out[d0][d1] = in[d0][d1] * rstd[d1] * weight[d0]
  %empty_out = tensor.empty() : tensor<4096x2048xf16>
  %norm = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d1)>,
                     affine_map<(d0, d1) -> (d0)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%in, %rstd, %w : tensor<4096x2048xf16>, tensor<2048xf32>, tensor<4096xf16>)
    outs(%empty_out : tensor<4096x2048xf16>) {
  ^bb0(%x: f16, %r: f32, %wt: f16, %out: f16):
    %xf = arith.extf %x : f16 to f32
    %normed = arith.mulf %xf, %r : f32
    %wf = arith.extf %wt : f16 to f32
    %scaled = arith.mulf %normed, %wf : f32
    %result_val = arith.truncf %scaled : f32 to f16
    linalg.yield %result_val : f16
  } -> tensor<4096x2048xf16>

  iree_tensor_ext.dispatch.tensor.store %norm, %result, offsets = [0, 0], sizes = [4096, 2048], strides = [1, 1] : tensor<4096x2048xf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4096x2048xf16>>
  return
}
