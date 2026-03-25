// LayerNorm over last 2 dims: tensor<8x512x10x128xf16>, reduce over d2,d3
// Pattern: Fused layernorm from Apex. Multiple reduction dimensions.
// Output: tensor<8x512x10x128xf16>, gamma tensor<10x128xf16>, beta tensor<10x128xf16>
// Normalize over the last 2 dims (10*128 = 1280 elements per reduction group)
//
// == Optimal Configuration ==
// reduction_size = 10*128 = 1280, parallel_size = 8*512 = 4096
// f16 input/output, f32 accumulation
// threads_per_row = 256 (ceil(1280/8) = 160, next pow2 = 256)
//   With 256 threads * 8 vec_elems = 2048 capacity, 1280 needs masking.
// vector_width = 8 (f16)
// elements_per_thread = 5 (1280 / 256 = 5, not a clean vector multiple)
//   Actually: first 160 threads handle 8 elems each, but 160*8=1280.
//   With 256 threads: threads 0-159 active, 160-255 masked.
// rows_per_block = 1 (256 threads for one reduction group)
// workgroup_size = 256
// num_workgroups = 4096 (8 * 512)
// reduction_strategy: 256 threads -> 8 warps, shuffle + shmem.
//   But many threads masked (only 160 of 256 active).
// coalescing: d3 (size 128) is innermost. Each row of the 10x128 reduction
//   space has 128 contiguous f16 = 256 bytes. Good coalescing within d3.
//   d2 stride = 128 elements, so sequential d2 iterations are also contiguous.
// masking: 1280 elements, 256 threads -> threads 160-255 idle.

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @layernorm_3d() {
  %cst = arith.constant 0.000000e+00 : f32
  %cst_eps = arith.constant 9.99999974E-6 : f32
  %cst_n = arith.constant 1.280000e+03 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x512x10x128xf16>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<10x128xf16>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<10x128xf16>>
  %3 = hal.interface.binding.subspan layout(#pipeline_layout) binding(3) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8x512x10x128xf16>>
  %input = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [8, 512, 10, 128], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x512x10x128xf16>> -> tensor<8x512x10x128xf16>
  %gamma = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [10, 128], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<10x128xf16>> -> tensor<10x128xf16>
  %beta = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0, 0], sizes = [10, 128], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<10x128xf16>> -> tensor<10x128xf16>
  %empty_4d = tensor.empty() : tensor<8x512x10x128xf32>
  %empty_2d = tensor.empty() : tensor<8x512xf32>
  %empty_out = tensor.empty() : tensor<8x512x10x128xf16>

  // Op 1: extf
  %extended = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    ins(%input : tensor<8x512x10x128xf16>) outs(%empty_4d : tensor<8x512x10x128xf32>) {
  ^bb0(%in: f16, %out: f32):
    %v = arith.extf %in : f16 to f32
    linalg.yield %v : f32
  } -> tensor<8x512x10x128xf32>

  // Op 2: fill for mean
  %acc_init = linalg.fill ins(%cst : f32) outs(%empty_2d : tensor<8x512xf32>) -> tensor<8x512xf32>

  // Op 3: sum over d2, d3 for mean
  %sum = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel", "reduction", "reduction"]}
    ins(%extended : tensor<8x512x10x128xf32>) outs(%acc_init : tensor<8x512xf32>) {
  ^bb0(%in: f32, %out: f32):
    %s = arith.addf %in, %out : f32
    linalg.yield %s : f32
  } -> tensor<8x512xf32>

  // Op 4: mean = sum / N
  %mean = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%sum : tensor<8x512xf32>) outs(%empty_2d : tensor<8x512xf32>) {
  ^bb0(%in: f32, %out: f32):
    %m = arith.divf %in, %cst_n : f32
    linalg.yield %m : f32
  } -> tensor<8x512xf32>

  // Op 5: fill for variance
  %var_init = linalg.fill ins(%cst : f32) outs(%empty_2d : tensor<8x512xf32>) -> tensor<8x512xf32>

  // Op 6: variance = sum((x - mean)^2) over d2, d3
  %var_sum = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel", "reduction", "reduction"]}
    ins(%extended, %mean : tensor<8x512x10x128xf32>, tensor<8x512xf32>) outs(%var_init : tensor<8x512xf32>) {
  ^bb0(%in: f32, %m: f32, %out: f32):
    %diff = arith.subf %in, %m : f32
    %sq = arith.mulf %diff, %diff : f32
    %s = arith.addf %sq, %out : f32
    linalg.yield %s : f32
  } -> tensor<8x512xf32>

  // Op 7: rstd = rsqrt(var/N + eps)
  %rstd = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%var_sum : tensor<8x512xf32>) outs(%empty_2d : tensor<8x512xf32>) {
  ^bb0(%in: f32, %out: f32):
    %v = arith.divf %in, %cst_n : f32
    %shifted = arith.addf %v, %cst_eps : f32
    %r = math.rsqrt %shifted : f32
    linalg.yield %r : f32
  } -> tensor<8x512xf32>

  // Op 8: normalize: (x - mean) * rstd * gamma + beta
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1)>,
                     affine_map<(d0, d1, d2, d3) -> (d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    ins(%input, %mean, %rstd, %gamma, %beta : tensor<8x512x10x128xf16>, tensor<8x512xf32>, tensor<8x512xf32>, tensor<10x128xf16>, tensor<10x128xf16>) outs(%empty_out : tensor<8x512x10x128xf16>) {
  ^bb0(%in: f16, %m: f32, %rs: f32, %g: f16, %b: f16, %out: f16):
    %xf = arith.extf %in : f16 to f32
    %gf = arith.extf %g : f16 to f32
    %bf = arith.extf %b : f16 to f32
    %diff = arith.subf %xf, %m : f32
    %normed = arith.mulf %diff, %rs : f32
    %scaled = arith.mulf %normed, %gf : f32
    %biased = arith.addf %scaled, %bf : f32
    %r = arith.truncf %biased : f32 to f16
    linalg.yield %r : f16
  } -> tensor<8x512x10x128xf16>

  iree_tensor_ext.dispatch.tensor.store %result, %3, offsets = [0, 0, 0, 0], sizes = [8, 512, 10, 128], strides = [1, 1, 1, 1] : tensor<8x512x10x128xf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8x512x10x128xf16>>
  return
}
