// LayerNorm with transposed output: normalize along d1 but write output transposed.
// Input: tensor<2048x4096xf16>, Output: tensor<4096x2048xf32>
// Reduction is along d1 (contiguous, inner), output stores as (d1, d0).
// This is the v1 Example 07 pattern -- the hardest coalescing case for norms.
//
// Source pattern: Apex contrib LayerNorm Example 07. v1 dropped output coalescing
// here. The correct solution is shared memory promotion for the output.
//
// == Optimal Configuration ==
// Block size: 256 threads
// threads_per_row: 256 (all on one row's reduction)
// Vector width: 8 (f16 input), 4 (f32 output)
// Elements per thread: 4096/256 = 16 input elements, 16 output elements
// Reduction: warp shuffle (5 steps) + cross-warp smem (8 warps -> 3 more steps)
// Workgroups: 2048 (one per row)
//
// Coalescing analysis:
//   INPUT: d1 is contiguous, threads map to d1 -> COALESCED
//   OUTPUT: stored as (d1, d0). Output contiguous dim is d0 (size 2048).
//     But each workgroup only writes one d0 value (its row). So output writes
//     are strided with stride 2048 between consecutive d1 elements within a WG.
//     This means output writes are NOT coalesced.
//
// Solution: Cache the normalized row in shared memory (4096 * sizeof(f32) = 16KB),
// then write to global memory with a transposed thread mapping.
// Alternative: Accept non-coalesced output writes (sometimes acceptable for norms
// since the reduction dominates).

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @layernorm_transpose_output() {
  %cst = arith.constant 0.000000e+00 : f32
  %cst_n = arith.constant 4096.0 : f32
  %cst_eps = arith.constant 1.0e-05 : f32
  %c0 = arith.constant 0 : index
  %input = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048x4096xf16>>
  %weight = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096xf16>>
  %output = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4096x2048xf32>>
  %in = iree_tensor_ext.dispatch.tensor.load %input, offsets = [0, 0], sizes = [2048, 4096], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048x4096xf16>> -> tensor<2048x4096xf16>
  %w = iree_tensor_ext.dispatch.tensor.load %weight, offsets = [0], sizes = [4096], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096xf16>> -> tensor<4096xf16>
  %empty_row = tensor.empty() : tensor<2048xf32>
  %empty_full = tensor.empty() : tensor<2048x4096xf32>
  %empty_out = tensor.empty() : tensor<4096x2048xf32>

  // Op 1: extf input to f32
  %ext = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%in : tensor<2048x4096xf16>) outs(%empty_full : tensor<2048x4096xf32>) {
  ^bb0(%a: f16, %b: f32):
    %v = arith.extf %a : f16 to f32
    linalg.yield %v : f32
  } -> tensor<2048x4096xf32>

  // Op 2: mean reduction
  %fill0 = linalg.fill ins(%cst : f32) outs(%empty_row : tensor<2048xf32>) -> tensor<2048xf32>
  %mean_sum = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]}
    ins(%ext : tensor<2048x4096xf32>) outs(%fill0 : tensor<2048xf32>) {
  ^bb0(%a: f32, %b: f32):
    %v = arith.addf %a, %b : f32
    linalg.yield %v : f32
  } -> tensor<2048xf32>

  // Op 3: mean = sum / N
  %mean = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>,
                     affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]}
    ins(%mean_sum : tensor<2048xf32>) outs(%empty_row : tensor<2048xf32>) {
  ^bb0(%a: f32, %b: f32):
    %v = arith.divf %a, %cst_n : f32
    linalg.yield %v : f32
  } -> tensor<2048xf32>

  // Op 4: variance reduction
  %fill1 = linalg.fill ins(%cst : f32) outs(%empty_row : tensor<2048xf32>) -> tensor<2048xf32>
  %var_sum = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>,
                     affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]}
    ins(%ext, %mean : tensor<2048x4096xf32>, tensor<2048xf32>) outs(%fill1 : tensor<2048xf32>) {
  ^bb0(%x: f32, %m: f32, %acc: f32):
    %sub = arith.subf %x, %m : f32
    %sq = arith.mulf %sub, %sub : f32
    %add = arith.addf %sq, %acc : f32
    linalg.yield %add : f32
  } -> tensor<2048xf32>

  // Op 5: normalize + weight + TRANSPOSE OUTPUT
  // Input indexing: (d0, d1) -> normal
  // Output indexing: (d0, d1) -> (d1, d0)  -- TRANSPOSED
  %norm = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>,
                     affine_map<(d0, d1) -> (d0)>,
                     affine_map<(d0, d1) -> (d1)>,
                     affine_map<(d0, d1) -> (d1, d0)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%ext, %mean, %var_sum, %w : tensor<2048x4096xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<4096xf16>)
    outs(%empty_out : tensor<4096x2048xf32>) {
  ^bb0(%x: f32, %m: f32, %v: f32, %wt: f16, %out: f32):
    %var = arith.divf %v, %cst_n : f32
    %eps = arith.addf %var, %cst_eps : f32
    %rstd = math.rsqrt %eps : f32
    %centered = arith.subf %x, %m : f32
    %normed = arith.mulf %centered, %rstd : f32
    %wf = arith.extf %wt : f16 to f32
    %scaled = arith.mulf %normed, %wf : f32
    linalg.yield %scaled : f32
  } -> tensor<4096x2048xf32>

  iree_tensor_ext.dispatch.tensor.store %norm, %output, offsets = [0, 0], sizes = [4096, 2048], strides = [1, 1] : tensor<4096x2048xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4096x2048xf32>>
  return
}
