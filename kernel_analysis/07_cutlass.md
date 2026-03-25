# CUTLASS: Memory-Bound Kernel Analysis

---

## CUTLASS Memory-Bound Kernel Pattern Analysis

### 1. REDUCTION KERNELS

#### 1a. ReduceSplitK -- Split-K Reduction Kernel
**File:** `/home/kunwar/Work/kernel_libs/cutlass/include/cutlass/reduction/kernel/reduce_split_k.h`

**Thread Block Configuration (lines 55-68, 134-147):**
- Template params: `Shape_` (MatrixShape), `OutputOp_`, `ReductionOp_`, `PartitionsPerStage = 4`
- `kElementsPerAccess = OutputOp::kCount` (derived from epilogue op, typically 128/sizeof_bits)
- Block shape: `dim3(Shape::kColumn / kElementsPerAccess, Shape::kRow)` -- threads in x cover columns via vectorized access, threads in y cover rows
- Grid shape: `dim3(ceil(rows/Shape::kRow), ceil(cols/Shape::kColumn))`

**Vectorized Access (lines 78-80):**
```cpp
using FragmentWorkspace = AlignedArray<ElementWorkspace, kElementsPerAccess>;
using FragmentAccumulator = Array<ElementAccumulator, kElementsPerAccess>;
using FragmentOutput = AlignedArray<ElementOutput, kElementsPerAccess>;
```
Loads/stores use `AlignedArray` cast as pointers for vectorized memory access via `*reinterpret_cast<FragmentWorkspace const *>(ptr)`.

**Reduction Strategy (lines 193-210):**
- Software pipelining with `kPartitionsPerStage = 4` -- loads a batch of 4 fragments, then reduces all 4
- Outer loop (NO_UNROLL) over partitions in steps of `kPartitionsPerStage`
- Inner loops (UNROLL) for load and accumulate
- Each thread independently reduces its vector position across all K partitions -- no cross-thread communication needed

**Key Constants:**
- `kPartitionsPerStage = 4` (batch size for software pipelining)
- `kElementsPerAccess` = typically 128/element_bits (e.g., 8 for f16, 4 for f32)

---

#### 1b. TensorReductionAffineContiguous -- General Tensor Reduction (Contiguous Dim)
**File:** `/home/kunwar/Work/kernel_libs/cutlass/include/cutlass/reduction/kernel/tensor_reduce_affine_contiguous.h`

**Thread Block Configuration (lines 63-64, 177-179):**
- `Threads = 256` (default)
- `BatchSize = 4` (elements loaded per software-pipeline batch)
- `VectorLength = 1` (default, but can be increased for contiguous reductions)
- SharedStorage: `Array<ElementCompute, kThreads * kVectorLength>` -- 256 * VectorLength elements

**Vectorized Access (lines 177-179):**
```cpp
using ComputeFragment = Array<ElementCompute, VectorLength>;
using SourceFragment = AlignedArray<ElementSource, VectorLength>;
using OutputFragment = AlignedArray<ElementOutput, VectorLength>;
```

**Reduction Strategy (lines 245-364):**
Three-level decomposition:
1. **Thread-level**: Each thread accumulates over its assigned portion of the inner dimension (lines 282-315), loading batches of `kBatchSize=4` fragments with software pipelining
2. **Vector-to-scalar**: Thread reduces its vector accumulator to a scalar (lines 320-326)
3. **Block-level**: Tree reduction in shared memory using `__syncthreads()` barriers (lines 334-360). Halves active threads at each step: `while (thread_count > 1) { thread_count /= 2; ... }`

**Grid/Threadblock Planning** (device-level file `/home/kunwar/Work/kernel_libs/cutlass/include/cutlass/reduction/device/tensor_reduce_affine_contiguous.h`, lines 142-231):
- `target_threadblock_count = 128` (default target occupancy)
- Priority 1: Assign threadblocks to outer indices (grid.y = min(outer_count, 128))
- Priority 2: Parallelize inner dimension across grid.z if inner_count > threads_per_block
- Uses `reshape_pow2()` to find optimal thread count when inner dimension is small
- Two-phase approach: if `grid.z > 1`, uses a device workspace + final reduction kernel

**Key Constants:**
- `kThreads = 256`
- `kBatchSize = 4`
- `target_threadblock_count = 128`
- `kVectorLength = 1` (default)

---

#### 1c. TensorReductionAffineStrided -- Strided Dimension Reduction
**File:** `/home/kunwar/Work/kernel_libs/cutlass/include/cutlass/reduction/kernel/tensor_reduce_affine_strided.h`

Same template parameters and defaults as contiguous (Threads=256, BatchSize=4, VectorLength=1). This variant handles the case where the reduction dimension is NOT the contiguous dimension -- no vectorization benefit on the reduction dimension itself. Uses `FastDivmodU64` for efficient coordinate decomposition.

---

### 2. SOFTMAX KERNELS

#### 2a. ApplySoftmaxFinalReduction -- Cross-Block Reduction for Softmax
**File:** `/home/kunwar/Work/kernel_libs/cutlass/include/cutlass/reduction/kernel/reduce_softmax_final.h`

**Thread Block Configuration (lines 183-205):**
- Launch: `blockDim.x = bdim` (variable), `blockIdx.x` partitions M dimension
- Each thread processes one row (or strided rows for grouped problems)

**Reduction Strategy (lines 222-258):**
Two-pass over threadblock-level partial results:
1. **Pass 1**: Find global max across all threadblocks' partial max values
2. **Pass 2**: Re-read partial max + partial sums, compute corrected sum as `sum += partial_sum * exp(partial_max - global_max)`
3. Final output: stores `max_val` and `inv_sum = 1/sum_val`

Uses `cutlass::arch::global_load` for scalar loads and `cutlass::fast_exp` for exponentials.

---

#### 2b. ApplySoftmax -- The Softmax Application Kernel
**File:** `/home/kunwar/Work/kernel_libs/cutlass/examples/35_gemm_softmax/gemm_with_softmax.h`, lines 76-286

**Thread Block Configuration (lines 82-83, 202-212):**
- `ApplyShape = MatrixShape<1, 1024>` (default: 1 row x 1024 columns per block)
- `kAlignment = Alignment` (template param, default = `128 / sizeof_bits<ElementC>`, e.g., 8 for f16)
- `threadIdx.y` indexes row within block (ApplyShape::kRow threads)
- `threadIdx.x * kAlignment` indexes column start

**Vectorized Access (lines 202, 226-228):**
```cpp
using AccessTypeD = AlignedArray<ElementD, kAlignment>;      // e.g., AlignedArray<half, 8>
using AccessTypeSoft = AlignedArray<ElementSoft, kAlignment>;
```
Uses `arch::global_load<AccessTypeD, sizeof(AccessTypeD)>` and `arch::global_store` for vectorized 128-bit transfers.

**Key Computation (line 275):**
```cpp
result = mul(exponential(minus(convert_soft_compute(fetch), convert_norm(norm))), convert_sum(inv_sum));
```
Pattern: `softmax(x) = exp(x - max) * inv_sum` -- entirely elementwise after reduction.

---

#### 2c. EpilogueVisitorSoftmax -- Fused In-Epilogue Softmax
**File:** `/home/kunwar/Work/kernel_libs/cutlass/include/cutlass/epilogue/threadblock/epilogue_visitor_with_softmax.h`

**Key Constants (lines 69-92):**
- `kThreadCount = ThreadCount` (from GEMM config)
- `kIterations = OutputTileIterator::kIterations`
- `kElementsPerAccess = OutputTileIterator::kElementsPerAccess`
- `kThreadsPerRow = OutputTileIterator::ThreadMap::Detail::kAccessWidth`

**Warp-Level Reduction (lines 433-452):**
```cpp
// Warp-level butterfly reduction for sum and max
for (int i = half_thread_in_row; i > 0; i >>= 1) {
    ElementSoftmaxCompute tmp = __shfl_xor_sync(0xFFFFFFFF, sum_, i);
    sum_ += tmp;
}
```
Uses `__shfl_xor_sync` for warp-level butterfly reductions -- avoids shared memory.

**Online Softmax Algorithm (lines 345-363):**
Implements the "online" softmax from the FlashAttention paper (referenced at line 356: `https://arxiv.org/pdf/2205.14135v1.pdf`):
```
S* = S* * updater + sum_row(P')    where updater = exp(M* - M_row)
```
This computes softmax incrementally across columns within each row as the epilogue iterates over tiles.

---

#### 2d. FMHA CollectiveSoftmax -- Hopper-Era Softmax
**File:** `/home/kunwar/Work/kernel_libs/cutlass/examples/88_hopper_fmha/collective/fmha_collective_softmax.hpp`

**Reduction Strategy (lines 98-134):**
- Uses CuTe tensors with `layout_acc_mn` for MxN layout of accumulator
- **Thread-level max**: Linear scan across N dimension per-M row (lines 100-109)
- **Cross-thread max**: Butterfly shuffle via `__shfl_xor_sync` using `reduction_target_n` strides (lines 111-119)
- **Softmax application**: `exp2(scale * x - scale * max)` using `exp2f` for speed (lines 121-130)
- **Sum reduction**: CuTe's `reduce(acc_qk_mn(i, _), cute::plus{})` (line 133)
- **Online correction** (step_interleave_begin, lines 137-187): Rescales PV accumulator by `exp2(old_max - new_max)` before accumulating new softmax values

Uses `exp2.approx.f16` PTX instruction for half-precision exp2 (line 67).

---

### 3. EPILOGUE PATTERNS

#### 3a. LinearCombination -- Core Elementwise Epilogue
**File:** `/home/kunwar/Work/kernel_libs/cutlass/include/cutlass/epilogue/thread/linear_combination.h`

**Key Template Parameters (lines 57-67):**
- `Count`: Elements per operation = typically `128/sizeof_bits<ElementOutput>` (comment at line 60-61)
  - f16: Count = 8
  - f32: Count = 4
  - f64: Count = 2
  - int8: Count = 16
- All operations work on `Array<T, kCount>` fragments

**Computation Pattern (lines 219-250):**
```
D = alpha * accumulator + beta * source
```
Using `NumericArrayConverter` for type conversion, `multiplies<FragmentCompute>`, and `multiply_add<FragmentCompute>`.

---

#### 3b. LinearCombinationBiasElementwise -- Bias + Activation Fusion
**File:** `/home/kunwar/Work/kernel_libs/cutlass/include/cutlass/epilogue/thread/linear_combination_bias_elementwise.h`

**Pattern (lines 97-160):**
```
Z = ElementwiseOp(alpha * Acc + beta * C + bias)
T = Z (optional store)
```
Where `ElementwiseOp` can be any activation (ReLU, GELU, SiLU, etc.)

`kElementsPerAccess` matches LinearCombination (128/element_bits).

---

#### 3c. LinearCombinationResidualBlock -- Residual + Activation Fusion
**File:** `/home/kunwar/Work/kernel_libs/cutlass/include/cutlass/epilogue/thread/linear_combination_residual_block.h`

**Pattern (line 49):**
```
Output = UnaryOp(BinaryOp(BinaryOp(ActivationOp(TensorOp(X) + bias), residual1), residual2))
```
`kIsHeavy = true` flag (line 85) -- affects loop unrolling decisions in the epilogue.

---

#### 3d. Activation Functions
**File:** `/home/kunwar/Work/kernel_libs/cutlass/include/cutlass/epilogue/thread/activation.h`

All activations have `Array<T, N>` specializations. Key activations and their `kIsHeavy` flag:
- `Identity<T>`: `kIsHeavy = false`
- `ReLu<T>`: `kIsHeavy = false` (just a max)
- `Scale<T>`: `kIsHeavy = false`
- GELU, SiLU, Sigmoid: `kIsHeavy = true` (involve exp/tanh)
- `Clamp<T>`: clamp between bounds

---

### 4. MEMORY ACCESS ABSTRACTIONS

#### 4a. AlignedArray
**File:** `/home/kunwar/Work/kernel_libs/cutlass/include/cutlass/array.h`, lines 2858-2870

```cpp
template <typename T, int N, int Alignment = (sizeof_bits<T>::value * N + 7) / 8>
class alignas(Alignment) AlignedArray: public Array<T, N> {};
```
Default alignment = total byte size of the array. This is the key abstraction for vectorized loads/stores. When used as `AlignedArray<half_t, 8>`, it generates 16-byte (128-bit) aligned loads.

#### 4b. Thread-Level Reduce with SIMD
**File:** `/home/kunwar/Work/kernel_libs/cutlass/include/cutlass/reduction/thread/reduce.h`

The `half_t` specialization (lines 90-157) uses `__hadd2` for pairwise half-precision addition, processing 2 elements per cycle via half2 intrinsics. Falls back to scalar loop on pre-SM60.

---

### 5. OUTPUT TILE THREAD MAP -- Coalescing Strategy

#### 5a. OutputTileOptimalThreadMap
**File:** `/home/kunwar/Work/kernel_libs/cutlass/include/cutlass/epilogue/threadblock/output_tile_thread_map.h`

**Key Constants (lines 225-227, 291-296):**
- `kMemoryAccessSize = 256` bytes (preferred access size per warp, line 225)
- `kWarpSize = 32`
- `kTargetMemoryAccessWidth = 256 / (kElementsPerAccess * kElementSize / 8)` (lines 235-236)

**Coalescing Algorithm (lines 223-266):**
The `RowArrangement` metaprogram distributes threads across rows and columns to achieve 256-byte coalesced accesses:
- `kAccessWidth = min(ShapeWidth, min(32, 256/(elementsPerAccess * elementBytes)))` -- how many threads read adjacent columns
- `kAccessRows = min(Shape::kRow, 32/kAccessWidth)` -- threads that span multiple rows
- For f16 with 8 elements/access (16 bytes): `kTargetMemoryAccessWidth = 256/16 = 16` threads reading adjacent columns

---

#### 5b. DefaultThreadMapTensorOp
**File:** `/home/kunwar/Work/kernel_libs/cutlass/include/cutlass/epilogue/threadblock/default_thread_map_tensor_op.h`

**Key Constant (line 73):** `kTensorOpRows = 8` -- tensor ops fundamentally operate on 8-row tiles.

The thread map shape is:
```cpp
OutputTileShape<ThreadblockShape::kN, 8, WarpCount::kM, 1, 1>  // Column, Row, Group, Cluster, Tile
```

---

#### 5c. PitchLinearStripminedThreadMap
**File:** `/home/kunwar/Work/kernel_libs/cutlass/include/cutlass/transform/pitch_linear_thread_map.h`

Strip-mines a tile across threads: first along contiguous dimension, then strided. Key formula (lines 103-114):
- If `Threads >= ShapeVec::kContiguous`: each thread does 1 column iteration, multiple row iterations
- Else: each thread does multiple column iterations

---

### 6. KEY EPILOGUE REDUCTION: EpilogueGemmKReduction
**File:** `/home/kunwar/Work/kernel_libs/cutlass/include/cutlass/epilogue/threadblock/epilogue_gemm_k_reduction.h`

**Warp-Level Reduction (lines 157-159):**
```cpp
sum[i] += __shfl_xor_sync(0xffffffff, sum[i], 1);
sum[i] += __shfl_xor_sync(0xffffffff, sum[i], 2);
```
Two butterfly reduction steps -- reduces across 4 threads (groups of 4 within a warp column).

`kIterations = kWarpSize / 8` -- for a warp size of 32: 4 iterations per thread.

---

### SUMMARY OF HARD CONSTANTS ACROSS CUTLASS MEMORY-BOUND KERNELS

| Parameter | Value | Source |
|---|---|---|
| Default thread count for reductions | **256** | `tensor_reduce_affine_contiguous.h:63` |
| Software pipeline batch size | **4** | `reduce_split_k.h:59`, `tensor_reduce_affine_contiguous.h:64` |
| Target threadblock count | **128** | `tensor_reduce_affine_contiguous.h:144` |
| Preferred access size (bytes) | **256** | `output_tile_thread_map.h:225` |
| Warp size | **32** | everywhere |
| TensorOp row granularity | **8** | `default_thread_map_tensor_op.h:73` |
| Default vector width (128-bit line) | **128/sizeof_bits<Element>** | throughout (f16=8, f32=4, f64=2, i8=16) |
| AlignedArray default alignment | **sizeof(T)*N bytes** | `array.h:2865` |
| Softmax apply tile | **MatrixShape<1, 1024>** | `gemm_with_softmax.h:82` |
| Softmax alignment | **128/sizeof_bits<ElementC>** | `gemm_with_softmax.h:312` |