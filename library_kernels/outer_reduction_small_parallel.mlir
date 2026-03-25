// Outer reduction with small parallel dim: tensor<8192x64xf32> -> tensor<64xf32>
// Reduce d0 (8192), parallel d1 (64). Like BatchNorm stats with 64 channels.
//
// Source pattern: MIOpen BatchNorm spatial (one WG per channel), Apex BN NCHW
// (blockIdx.x = channel). PyTorch batch_norm_collect_statistics_kernel.
//
// == Optimal Configuration ==
// Block size: 256 threads (1D, all on reduction since d1=64 is small)
// Grid: 64 workgroups (one per output element / channel)
// Vector width: 4 (f32)
// Elements per thread: 8192/256 = 32 elements along d0
//
// Thread mapping: Each workgroup handles one d1 position. All 256 threads
// cooperate on the d0 reduction. Thread t reads d0 = t, t+256, t+512, ...
//
// Memory access: Reads are STRIDED -- thread t reads input[t][channel].
// The stride between consecutive thread reads = d1 = 64 elements = 256 bytes.
// This means 32 consecutive threads (one warp) each read from addresses
// 256 bytes apart -- NOT coalesced (stride of 256B between adjacent threads).
// This is the fundamental problem with outer reductions on row-major data.
//
// Alternative: Could transpose the input first, or use a 2D thread block where
// threadIdx.x maps to a tile of d0 positions. But with d1=64, the number of
// output elements is small enough that 1-WG-per-output is the right approach
// despite the strided access.
//
// Reduction: warp shuffle (5 steps) + cross-warp smem (8 warps -> 3 steps).

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @outer_reduction_small_parallel() {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8192x64xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<64xf32>>
  %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [8192, 64], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8192x64xf32>> -> tensor<8192x64xf32>
  %3 = tensor.empty() : tensor<64xf32>
  %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<64xf32>) -> tensor<64xf32>
  %5 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d1)>],
    iterator_types = ["reduction", "parallel"]}
    ins(%2 : tensor<8192x64xf32>) outs(%4 : tensor<64xf32>) {
  ^bb0(%in: f32, %out: f32):
    %6 = arith.addf %in, %out : f32
    linalg.yield %6 : f32
  } -> tensor<64xf32>
  iree_tensor_ext.dispatch.tensor.store %5, %1, offsets = [0], sizes = [64], strides = [1] : tensor<64xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<64xf32>>
  return
}
