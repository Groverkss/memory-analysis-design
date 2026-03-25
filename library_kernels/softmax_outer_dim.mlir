// Softmax on tensor<4096x1024xf32>, reduce along d0 (outer dimension).
// Pattern: Column-wise softmax (spatial softmax from PyTorch). Outer
// reduction = strided memory access.
//
// == Optimal Configuration ==
// Block size: 256 threads (8 warps), organized as 2D: blockDim.x=32, blockDim.y=8
// Thread mapping:
//   threadIdx.x → d1 (parallel, spatial/column dimension, stride-1 access)
//   threadIdx.y → d0 (reduction dimension, strided access)
// Vector width: 1 for the reduction axis (strided), but threads along d1 are
//   contiguous so 32 consecutive threads form coalesced 128-byte loads.
// Elements per thread: 4096 / 8 = 512 reduction elements per thread (looped)
// Data fits in registers: NO — 512 x f32 = 2048 bytes/thread. Must stream.
// Multi-pass: YES — each thread loops over 512 elements along d0.
//   With blockDim.y=8, each pass covers 8 rows. ceil(4096/8) = 512 passes.
// Reduction strategy:
//   Level 1: Each thread accumulates its 512 elements sequentially (register).
//   Level 2: Reduce across threadIdx.y (8 threads) via shared memory.
//   No warp shuffle across the reduction dim because reduction threads are
//   in different warps (threadIdx.y varies, warps are 32 consecutive threadIdx).
//   Actually: thread (x,y) has linear ID = y*32+x. Threads with same x but
//   different y are in different warps, so must use shared memory.
// Number of workgroups: ceil(1024/32) = 32 workgroups
//   Each workgroup handles 32 columns and all 4096 rows.
// Coalescing:
//   Input read: threads with consecutive threadIdx.x read consecutive d1
//   elements in the same row → coalesced (32 x 4 bytes = 128 bytes).
//   But different threadIdx.y values read different rows → stride of 1024*4
//   bytes between them. For a given warp (32 threads with same y), reads
//   ARE coalesced. Different warps read different rows independently.
//   Output write: same pattern, coalesced within each warp.
// Softmax multi-stage:
//   Stage 1 (max): Each thread loops d0 in steps of 8, accumulates max.
//     After loop: 8 partial maxes per column reduced via smem → column max.
//   Stage 2 (sub+exp+sum): Loop again, reload, subtract max, exp, accumulate
//     sum. Store exp values. After loop: smem reduce → column sum.
//   Stage 3 (div): Loop again, load exp, divide by sum, store.
//   3 loads + 2 stores per element (can't fuse across reduction loops).

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @softmax_outer_dim() {
  %cst_min = arith.constant 0xFF800000 : f32
  %cst_zero = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index

  %input_binding = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x1024xf32>>
  %output_binding = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4096x1024xf32>>

  %input = iree_tensor_ext.dispatch.tensor.load %input_binding, offsets = [0, 0], sizes = [4096, 1024], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x1024xf32>> -> tensor<4096x1024xf32>

  // Op1: Max reduction along d0 (outer dim).
  %empty_max = tensor.empty() : tensor<1024xf32>
  %fill_max = linalg.fill ins(%cst_min : f32) outs(%empty_max : tensor<1024xf32>) -> tensor<1024xf32>
  %max = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d1)>],
    iterator_types = ["reduction", "parallel"]}
    ins(%input : tensor<4096x1024xf32>) outs(%fill_max : tensor<1024xf32>) {
  ^bb0(%in: f32, %out: f32):
    %0 = arith.maximumf %in, %out : f32
    linalg.yield %0 : f32
  } -> tensor<1024xf32>

  // Op2: Subtract max (broadcast along d0) + exp.
  %empty_exp = tensor.empty() : tensor<4096x1024xf32>
  %exp = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%input, %max : tensor<4096x1024xf32>, tensor<1024xf32>) outs(%empty_exp : tensor<4096x1024xf32>) {
  ^bb0(%in: f32, %mx: f32, %out: f32):
    %0 = arith.subf %in, %mx : f32
    %1 = math.exp %0 : f32
    linalg.yield %1 : f32
  } -> tensor<4096x1024xf32>

  // Op3: Sum reduction along d0 (outer dim).
  %empty_sum = tensor.empty() : tensor<1024xf32>
  %fill_sum = linalg.fill ins(%cst_zero : f32) outs(%empty_sum : tensor<1024xf32>) -> tensor<1024xf32>
  %sum = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d1)>],
    iterator_types = ["reduction", "parallel"]}
    ins(%exp : tensor<4096x1024xf32>) outs(%fill_sum : tensor<1024xf32>) {
  ^bb0(%in: f32, %out: f32):
    %0 = arith.addf %in, %out : f32
    linalg.yield %0 : f32
  } -> tensor<1024xf32>

  // Op4: Divide by sum (broadcast along d0).
  %empty_out = tensor.empty() : tensor<4096x1024xf32>
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%exp, %sum : tensor<4096x1024xf32>, tensor<1024xf32>) outs(%empty_out : tensor<4096x1024xf32>) {
  ^bb0(%in: f32, %s: f32, %out: f32):
    %0 = arith.divf %in, %s : f32
    linalg.yield %0 : f32
  } -> tensor<4096x1024xf32>

  iree_tensor_ext.dispatch.tensor.store %result, %output_binding, offsets = [0, 0], sizes = [4096, 1024], strides = [1, 1] : tensor<4096x1024xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4096x1024xf32>>
  return
}
