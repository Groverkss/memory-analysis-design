// Fused residual add + RMSNorm: y = rmsnorm(x + residual) * weight
// Pattern: FlashInfer/Liger-Kernel decoder layer fusion.
// 3 inputs (x, residual, weight), 2 outputs (normed, x+residual for next layer)
// tensor<2048x4096xf16> for x and residual, tensor<4096xf16> for weight
// Output: tensor<2048x4096xf16> (normed), tensor<2048x4096xf16> (x+residual)
//
// == Optimal Configuration ==
// hidden_dim=4096, batch=2048, f16 input/output, f32 accumulation
// threads_per_row = 256
// vector_width = 8 (f16)
// elements_per_thread = 16 (4096 / 256)
// rows_per_block = 1
// workgroup_size = 256
// num_workgroups = 2048
// reduction_strategy: 256 threads -> 8 warps, shuffle + shmem
// coalescing: d1 innermost, 256*8*2 = 4096 bytes per pass. Excellent.
// The fusion saves one full read+write of the intermediate (x+residual)
//   compared to separate add and rmsnorm kernels. This is the key
//   optimization: 2048*4096*2 = 16MB saved in global memory traffic.
// register pressure: 16 f16 (x) + 16 f16 (residual) + 16 f32 (sum) +
//   16 f32 (sq) = 160 bytes per thread. Comfortable.
// 2 outputs: normed result AND the residual sum (for the next layer's
//   residual connection). Both stored to separate output bindings.

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @fused_residual_rms_norm() {
  %cst = arith.constant 0.000000e+00 : f32
  %cst_eps = arith.constant 9.99999974E-6 : f32
  %cst_n = arith.constant 4.096000e+03 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048x4096xf16>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048x4096xf16>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096xf16>>
  %3 = hal.interface.binding.subspan layout(#pipeline_layout) binding(3) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2048x4096xf16>>
  %4 = hal.interface.binding.subspan layout(#pipeline_layout) binding(4) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2048x4096xf16>>
  %x = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [2048, 4096], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048x4096xf16>> -> tensor<2048x4096xf16>
  %residual = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [2048, 4096], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048x4096xf16>> -> tensor<2048x4096xf16>
  %weight = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0], sizes = [4096], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096xf16>> -> tensor<4096xf16>
  %empty_2d_f16 = tensor.empty() : tensor<2048x4096xf16>
  %empty_2d_f32 = tensor.empty() : tensor<2048x4096xf32>
  %empty_1d = tensor.empty() : tensor<2048xf32>
  %empty_out = tensor.empty() : tensor<2048x4096xf16>

  // Op 1: residual add (x + residual) -> store as second output
  %added = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%x, %residual : tensor<2048x4096xf16>, tensor<2048x4096xf16>) outs(%empty_2d_f16 : tensor<2048x4096xf16>) {
  ^bb0(%a: f16, %b: f16, %out: f16):
    %s = arith.addf %a, %b : f16
    linalg.yield %s : f16
  } -> tensor<2048x4096xf16>

  // Op 2: extf + square of the sum
  %squared = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%added : tensor<2048x4096xf16>) outs(%empty_2d_f32 : tensor<2048x4096xf32>) {
  ^bb0(%in: f16, %out: f32):
    %v = arith.extf %in : f16 to f32
    %sq = arith.mulf %v, %v : f32
    linalg.yield %sq : f32
  } -> tensor<2048x4096xf32>

  // Op 3: fill for reduction
  %acc_init = linalg.fill ins(%cst : f32) outs(%empty_1d : tensor<2048xf32>) -> tensor<2048xf32>

  // Op 4: reduce sum of squares
  %sum_sq = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]}
    ins(%squared : tensor<2048x4096xf32>) outs(%acc_init : tensor<2048xf32>) {
  ^bb0(%in: f32, %out: f32):
    %s = arith.addf %in, %out : f32
    linalg.yield %s : f32
  } -> tensor<2048xf32>

  // Op 5: rsqrt(mean + eps)
  %rrms = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>,
                     affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]}
    ins(%sum_sq : tensor<2048xf32>) outs(%empty_1d : tensor<2048xf32>) {
  ^bb0(%in: f32, %out: f32):
    %mean = arith.divf %in, %cst_n : f32
    %shifted = arith.addf %mean, %cst_eps : f32
    %r = math.rsqrt %shifted : f32
    linalg.yield %r : f32
  } -> tensor<2048xf32>

  // Op 6: normalize + weight
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>,
                     affine_map<(d0, d1) -> (d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%added, %rrms, %weight : tensor<2048x4096xf16>, tensor<2048xf32>, tensor<4096xf16>) outs(%empty_out : tensor<2048x4096xf16>) {
  ^bb0(%in: f16, %scale: f32, %w: f16, %out: f16):
    %xf = arith.extf %in : f16 to f32
    %wf = arith.extf %w : f16 to f32
    %normed = arith.mulf %xf, %scale : f32
    %weighted = arith.mulf %normed, %wf : f32
    %r = arith.truncf %weighted : f32 to f16
    linalg.yield %r : f16
  } -> tensor<2048x4096xf16>

  // Store both outputs: normed result and residual sum
  iree_tensor_ext.dispatch.tensor.store %result, %3, offsets = [0, 0], sizes = [2048, 4096], strides = [1, 1] : tensor<2048x4096xf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2048x4096xf16>>
  iree_tensor_ext.dispatch.tensor.store %added, %4, offsets = [0, 0], sizes = [2048, 4096], strides = [1, 1] : tensor<2048x4096xf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2048x4096xf16>>
  return
}
