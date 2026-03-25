# ThunderKittens: Memory-Bound Kernel Analysis

---

# ThunderKittens Exhaustive Kernel Analysis

## 1. Core Architecture & Type System

### Fundamental Constants
**File**: `/home/kunwar/Work/kernel_libs/ThunderKittens/include/common/util.cuh`

```cpp
constexpr int BASE_TILE_DIM = 16;                          // line 20
template<typename T> constexpr int TILE_COL_DIM = sizeof(T) == 1 ? 32 : 16;  // line 21
template<typename T> constexpr int TILE_ROW_DIM = 16;      // line 22
constexpr int WARP_THREADS{32};                             // line 30
constexpr int WARPGROUP_THREADS{128};                       // line 34
constexpr int WARPGROUP_WARPS{4};                           // line 39
constexpr int MAX_SHARED_MEMORY = 227 * 1024;  // Hopper/Blackwell (line 58)
constexpr int MAX_SHARED_MEMORY = 164 * 1024;  // Ampere (line 60)
```

The entire library is built on a **16x16 base tile** (the hardware MMA tile). All larger tiles are composed of multiples of this.

### Register Tile (`rt`) - The Core Compute Unit
**File**: `/home/kunwar/Work/kernel_libs/ThunderKittens/include/types/register/rt_base.cuh`

Each 16x16 base tile (`rt_base`) has:
- **4 packed values per thread** (`packed_per_thread = 4`, line 75)
- Each packed value is a `float2` (for fp32) or `bf16_2` (for bf16)
- So each thread holds **8 elements** of a 16x16 tile (256 elements / 32 threads = 8, line 73)
- **4 registers per thread** for fp32 tiles, **4 registers** for bf16 tiles (line 76)

The larger `rt<T, rows, cols>` is a 2D array of `rt_base` subtiles:
```cpp
rt_base<T, layout> tiles[height][width];  // line 103, rt.cuh
```

So `rt_fl<16, 128>` is 1 row x 8 columns of 16x16 subtiles = 32 registers per thread.

### Shared Tile (`st`) - Swizzled Shared Memory Layout
**File**: `/home/kunwar/Work/kernel_libs/ThunderKittens/include/types/shared/st.cuh`

Key swizzling for bank conflict avoidance (lines 91-103):
```cpp
static constexpr int swizzle_bytes = _swizzle_bytes > 0 ? _swizzle_bytes : (
    sizeof(dtype) == 2 ? (
        (cols/TILE_COL_DIM<T>)%4 == 0 ? 128 :
        (cols/TILE_COL_DIM<T>)%2 == 0 ?  64 : 32
    ) : ...
);
```

The swizzle pattern XORs address bits (line 114):
```cpp
const int swizzle = ((addr % swizzle_repeat) >> 7) << 4;
return (T*)(addr ^ swizzle);
```
This is the standard NVIDIA shared memory swizzle for bank conflict-free access to tiles that are multiples of 128 bytes wide.

### Global Layout (`gl`) - 4D Tensor Descriptor
**File**: `/home/kunwar/Work/kernel_libs/ThunderKittens/include/types/global/gl.cuh`

```cpp
template<typename _T, int b, int d, int r, int c, typename... TMA_Types>
struct gl {  // line 113
```
4D layout: **batch x depth x rows x cols**. Dimensions can be compile-time fixed (positive int) or runtime (`-1`). TMA descriptors are generated per tile type passed as variadic template args.

---

## 2. Pipeline Templates (The "Prototype" Framework)

### LCF Template (Load-Compute-Finish)
**File**: `/home/kunwar/Work/kernel_libs/ThunderKittens/prototype/lcf/lcf.cuh`

This is the core execution template. Structure:
- **Producer warpgroup(s)**: Load data from global to shared memory
- **Consumer warpgroup(s)**: Compute on data in registers/shared memory
- Semaphore-based synchronization between producer/consumer

Default configuration constants from `/home/kunwar/Work/kernel_libs/ThunderKittens/prototype/common/templates.cuh` (lines 42-62):
```cpp
FLAG_GETTER(NUM_BLOCKS, 1)
FLAG_GETTER(NUM_CONSUMER_WARPS, 8)
FLAG_GETTER(NUM_PRODUCER_WARPS, 4)
FLAG_GETTER(INPUT_PIPE_STAGES, 1)
FLAG_GETTER(OUTPUT_PIPE_STAGES, 1)
```

So default is: **8 consumer warps (2 warpgroups) + 4 producer warps (1 warpgroup) = 12 warps = 384 threads**.

### LCSF Template (Load-Compute-Store-Finish)
**File**: `/home/kunwar/Work/kernel_libs/ThunderKittens/prototype/lcsf/lcsf.cuh`

Same as LCF but adds an output pipeline stage with explicit store phase. Used for kernels that need to stream output back (rotary, mamba2, fftconv).

---

## 3. Memory-Bound Kernel: LayerNorm

**File**: `/home/kunwar/Work/kernel_libs/ThunderKittens/kernels/layernorm/layernorm.cu`

### Tile/Block Configuration
- **2 warps** per block (`NUM_WORKERS = 2`, line 12)
- **64 threads** per block (line 13)
- Hard-coded `D = 1024` (model dimension, line 171/232)
- Shared memory vectors are `sv_bf<1024>` -- 1024-element bf16 vectors

### Thread-to-Data Mapping
- Each warp handles one sequence position at a time
- Each block handles 2 sequence positions (`n_per_tile = 2`, line 197)
- Grid: `dim3(N/2, B, 1)` -- each block does 2 rows of the (B, N, D) tensor

### Shared Memory Layout (lines 102-105)
```cpp
vec_smem_1xD (&x_s)           [2][NUM_WORKERS]  // double-buffered, per-warp
vec_smem_1xD (&residual_s)    [2][NUM_WORKERS]  // double-buffered, per-warp
vec_smem_1xD (&norm_weight_s)                    // shared across warps
vec_smem_1xD (&norm_bias_s)                      // shared across warps
```
Total shared memory: 25480 bytes (line 201).

### Memory Access Pattern
- **Double-buffered async loads** (tic/toc pattern, lines 108, 119-131)
- `warp::load_async` for input x and residual
- Warp 0 pre-loads norm weights and bias (lines 111-114)
- Each iteration: load next pair while computing current

### Reduction Pattern (lines 141-147)
```cpp
warp::sum(mean, residual_s[warpid][tic]);  // full row sum
mean = mean / __float2bfloat16(d_model);    // scalar divide
warp::sub(residual_s, residual_s, mean);    // subtract mean
warp::mul(x_s, residual_s, residual_s);     // square
warp::sum(var, x_s);                        // variance sum
var = var / __float2bfloat16(d_model);
var = sqrt(var + 1e-05f);
```
All reductions are **warp-level** on shared vectors using `warp::sum`. No cross-warp reduction needed because each warp owns its row.

### Constants and Tuning
| Parameter | Value | Notes |
|-----------|-------|-------|
| d_model | 1024 | Hard-coded |
| warps/block | 2 | Minimal -- memory-bound |
| threads/block | 64 | |
| shared mem | 25480 bytes | |
| pipe stages | 2 (manual double-buffer) | |
| epsilon | 1e-05f | |

---

## 4. Memory-Bound Kernel: Rotary Embedding

**File**: `/home/kunwar/Work/kernel_libs/ThunderKittens/kernels/rotary/rotary.cu`

### Tile/Block Configuration
- Uses LCSF prototype template
- **8 consumer warps** (2 warpgroups) + 4 producer warps = **12 warps = 384 threads** (line 27)
- `INPUT_PIPE_STAGES=3`, `OUTPUT_PIPE_STAGES=3` (line 27)
- `NUM_BLOCKS=1`
- Supports headdim=64 or headdim=128

### Thread-to-Data Mapping
- Each consumer warp handles one 16-row block of sequence tokens
- Shared tile: `st_bf<16, headdim>` per warp (line 13)
- Grid: rows tiled by `8*16=128` rows per block, batches tiled

### Register File Usage (line 24)
```cpp
struct consumer_state { rt_fl<16, headdim/2> sin, cos; };
```
Sin/cos are **long-resident** in registers -- loaded once in `consumer::setup` (lines 73-75), never evicted. For headdim=128: `rt_fl<16, 64>` = 4 * 4 = 16 subtiles * 4 regs/subtile = 64 registers per sin/cos = **128 registers** permanently occupied.

### Memory Access Pattern
- **TMA async loads** for input sequence data
- Sin/cos loaded from global memory once via `warp::load` (non-async, blocking)
- Pure elementwise: load x, split into x1/x2 halves, multiply by sin/cos, add, store

### Computation Pattern (lines 83-103)
Manual register shuffling to split a `rt_fl<16, headdim>` into two `rt_fl<16, headdim/2>` halves by copying subtile data arrays directly. Then elementwise mul/add.

---

## 5. Compute-Bound (but memory-relevant) Kernel: MHA Forward (H100 Manual Pipeline)

**File**: `/home/kunwar/Work/kernel_libs/ThunderKittens/kernels/attention/mha_h100/mha_h100.cu`

### Tile/Block Configuration (lines 7-27)
```cpp
CONSUMER_WARPGROUPS = 3   // 12 consumer warps
PRODUCER_WARPGROUPS = 1   // 4 producer warps
NUM_WARPGROUPS = 4        // 16 total warps
```

Head dimension specializations:
| D | tile_width | qo_height | kv_height | pipeline stages |
|---|-----------|-----------|-----------|----------------|
| 64 | 64 | 64 (4*16) | 128 (8*16) | 4 |
| 128 | 128 | 64 (4*16) | 128 (8*16) | 2 |

### Shared Memory Layout (lines 67-71)
```cpp
q_tile  (&q_smem)[3]       // one per consumer warpgroup -- scratch
k_tile  (&k_smem)[stages]  // pipeline-buffered K
v_tile  (&v_smem)[stages]  // pipeline-buffered V
l_col_vec (&l_smem)[3]     // logsumexp output
o_tile reuses q_smem        // aliased output
```

For D=128: `st_bf<64, 128>` = 64*128*2 = 16384 bytes per Q tile, `st_bf<128, 128>` = 32768 bytes per KV tile. With 2 pipeline stages: 3*16384 + 2*32768 + 2*32768 = ~180KB shared memory.

### Register Budget (lines 106, 128)
- **Producer warpgroup**: `decrease_registers<32>()` -- 32 registers/thread
- **Consumer warpgroups**: `increase_registers<160>()` -- 160 registers/thread

Consumer registers hold (lines 130-134):
```cpp
rt_fl<16, kv_height>   att_block;        // 16x128 = 8 subtiles * 4 regs = 32 regs
rt_bf<16, kv_height>   att_block_mma;    // 16x128 bf16 = 16 regs
rt_fl<16, tile_width>  o_reg;            // 16x128 = 32 regs
col_vec<...> max_vec, norm_vec, etc;     // ~8 regs
```
Total: ~90 registers for data + overhead = ~160.

### Temperature Scaling Constants (lines 155-156)
```cpp
D == 64:  1.44269504089f * 0.125f       // = 1/sqrt(64) * log2(e)
D == 128: 1.44269504089f * 0.08838834764f // = 1/sqrt(128) * log2(e)
```

### Online Softmax + Flash Attention Pattern
Classic online softmax with rescaling:
1. Compute `att_block = Q @ K^T` via `warpgroup::mm_ABt`
2. Track running `max_vec` and `norm_vec`
3. Rescale `o_reg` by `exp2(max_old - max_new)` before accumulating new attention weights
4. Final: `o_reg /= norm_vec`

---

## 6. MHA Forward (LCF Prototype Template Version)

**File**: `/home/kunwar/Work/kernel_libs/ThunderKittens/kernels/attention/mha_h100_lcf/mha_h100_lcf.cu`

### Tile/Block Configuration (line 30)
```cpp
NUM_CONSUMER_WARPS = 12, NUM_WORKERS = 3, INPUT_PIPE_STAGES = 2
```
12 consumer warps = 3 warpgroups + default 4 producer warps = 16 warps = 512 threads.

### Key Difference: KV Tile Size is D-dependent (line 14)
```cpp
using kv_tile = st_bf<D==64?192:128, D>;
```
For D=64, the KV tile is **192x64** (12 rows of 16). For D=128, it's **128x128** (8 rows of 16). This tunes the inner loop granularity.

### Consumer State (lines 22-27)
```cpp
rt_fl<16, qo_tile::cols> o_reg;             // output accumulator
col_vec<rt_fl<16, kv_tile::rows>> max_vec, norm_vec;  // online softmax state
rt_fl<16, kv_tile::rows> att_block;          // attention scores
rt_bf<16, kv_tile::rows> att_block_mma;      // bf16 copy for MMA
```

### Softmax (line 65)
```cpp
constexpr float TEMPERATURE_SCALE = (D == 128) ? 0.08838834764f*1.44269504089f : 0.125f*1.44269504089f;
```

---

## 7. B300 Attention (Blackwell Architecture)

**File**: `/home/kunwar/Work/kernel_libs/ThunderKittens/kernels/attention/bf16_b300_mha_noncausal/bf16_b300_mha_noncausal.cu`

### Tile/Block Configuration (lines 7-30)
```cpp
Mb = 128, Nb = 128, Dqk = 192, Dvo = 128  // hard-coded asserts
CLUSTER_SIZE = 2       // 2-CTA clusters
NUM_SM = 148           // B300 SM count
NUM_PRODUCERS = 1, NUM_CORRECTORS = 1, NUM_SOFTMAXXERS = 2
TOTAL_WGS = 4 warpgroups = 16 warps = 512 threads
LOAD_STAGES = 3
DYNAMIC_SHARED_MEMORY = MAX_SHARED_MEMORY - 1024
```

### Novel Features
- Uses **tensor types** (`tt<float, Mb, Nb>`) for Blackwell tensor memory
- Uses **cluster-level TMA** with `tensor_allocator<1, CLUSTER_SIZE>`
- KV tiles are **split**: `k_tile = st_bf<Nb/2, Dqk>` (64x192), `v_tile = st_bf<Nb, Dvo/2>` (128x64) to fit shared memory constraints
- Persistent kernel: loops over tasks via `total_bids` with swizzled 2D indexing

---

## 8. Flux GEMM+GELU Fusion

**File**: `/home/kunwar/Work/kernel_libs/ThunderKittens/kernels/flux/flux_gelu.cu`

### Tile/Block Configuration (lines 42-65)
Parameterized by `BLOCK_M, BLOCK_N, BLOCK_K`:
```cpp
NUM_CONSUMER_WARPS = BLOCK_M/16
NUM_CONSUMER_WARPGROUPS = NUM_CONSUMER_WARPS/4
```

Benchmarked configs (lines 327-334):
| BLOCK_M | BLOCK_N | BLOCK_K | Use case |
|---------|---------|---------|----------|
| 192 | 192 | 64 | M=3072 (644 TFLOPs) |
| 128 | 192 | 64 | M=512 (509 TFLOPs) |
| 128 | 192 | 64 | M=256 (445 TFLOPs) |

### GELU Fusion Pattern (lines 114-121)
GELU is applied **in registers** after the GEMM accumulation, before store:
```cpp
f * 0.5f * (1.0f + fast_tanh(f * 0.79788456f * (1.f + f * f * 0.044715f)))
```
Uses `fast_tanh` via PTX `tanh.approx.f32` instruction (line 12).

### Bias Loading Pattern (lines 25-38)
Bias is loaded from shared vector into register tile via manual indexing:
```cpp
float2 tmp1 = __bfloat1622float2(*(bf16_2*)&bias.data[16*i + 0 + 2*(laneid()%4)]);
acc.tiles[0][i].data[0].x = tmp1.x;  // broadcast to all 4 rows
```
This maps the shared vector elements directly to the register tile layout matching the MMA output distribution.

---

## 9. Flux GEMM+Gate Fusion

**File**: `/home/kunwar/Work/kernel_libs/ThunderKittens/kernels/flux/flux_gate.cu`

### Additional Features over GELU
- Loads a **gate vector** and a **y matrix** in the finish phase
- Gate: `acc *= gate` (elementwise with broadcast vector)
- Y residual: `acc += y` (loaded from global via `warpgroup::load_async`)
- Register budget explicitly managed: `increase_registers<NUM_CONSUMER_WARPGROUPS==3?152:232>()` (line 100)

### Tile sizes dispatched by problem size (lines 448-473):
| Condition | M_tile | K_tile | N_tile |
|-----------|--------|--------|--------|
| M > 512 | 192 | 192 | 64 |
| M > 256 && K > 3072 | N/A | N/A | N/A |
| K > 3072 | 64 | 96 | 128 |
| else | 128 | 192 | 64 |

---

## 10. Based Linear Attention

**File**: `/home/kunwar/Work/kernel_libs/ThunderKittens/kernels/based/linear_attn.cu`

### Configuration
```cpp
D_QK = 16   // feature dimension for Q, K (line 10)
D_VO = 64   // feature dimension for V, O (line 11)
NUM_WORKERS = 4  // 4 warps (line 13)
```
Launch bounds: `__launch_bounds__(128, 2)` -- 2 blocks per SM.

### Shared Memory (lines 130-143)
```cpp
st_bf<64,16>  q_s[2], k_s[2]        // double-buffered, 4096 bytes each
st_bf<64,64>  v_s[2], v_s_2[2]      // 16384 bytes each (v_s_2 = duplicate to avoid WGMMA hazard)
st_bf<64,64>  o_s[2]                 // output staging
st_bf<64,16>  a1_trans_s             // KV state
st_bf<64,64>  a2_s[4]               // 32768 bytes
sv_fl<64>     a0_float               // cumulative sum accumulator
```
Total: ~98000 bytes (line 358).

### Key Pattern: Taylor-expanded attention
The kernel computes `1 + QK^T + (QK^T)^2/2` (lines 211-219) as a polynomial approximation of softmax, then applies causal masking.

### Cross-Warp Reduction (lines 54-69)
`accumulate_a0` uses `atomicAdd` to a shared float vector -- each warp reduces its rows, then atomically accumulates:
```cpp
atomicAdd(&a0_float[col], acc.x);
```

### Shuffle-based Slice Operations (lines 74-115)
`mul_slice_row` and `mul_slice_col` use `__shfl_sync` to broadcast specific columns/rows across a warp for efficient outer-product-like operations.

---

## 11. Mamba2 (SSM)

**File**: `/home/kunwar/Work/kernel_libs/ThunderKittens/kernels/mamba2/mamba2.cu`

### Configuration (line 51)
```cpp
NUM_CONSUMER_WARPS=8, OUTPUT_PIPE_STAGES=2, INPUT_PIPE_STAGES=2
PRODUCER_BARRIER_ARRIVALS=1, CONSUMER_BARRIER_ARRIVALS=2  // 2 warpgroups
```

### Tile Configuration
All tiles are 64x64 bf16:
```cpp
using q_tile = st_bf<64, 64>;
using k_tile = st_bf<64, 64>;
using v_tile = st_bf<64, 64>;
```

### Consumer State (lines 42-48)
```cpp
rt_fl<16, 64> o_reg, att_block, local_decay, kv;
rt_bf<16, 64> att_block_mma, q_reg, k_reg;
```
Two warpgroups handle 2 heads in parallel. The `kv` state is carried across iterations (initialized to zero, accumulated).

### Decay Computation
Hillis-Steele prefix sum in shared memory (lines 107-113) for computing cumulative decay, then exponentiated:
```cpp
warp::exp(args.state.local_decay, args.state.local_decay);
```

---

## 12. Hedgehog (Hybrid Attention)

**File**: `/home/kunwar/Work/kernel_libs/ThunderKittens/kernels/hedgehog/hedgehog.cu`

### Configuration
```cpp
NUM_WORKERS = 8 (warps)       // line 16
CHUNK_SIZE = 64               // line 48
ATTN_D = 128                  // line 49
ATTN_F = 128 (feature dim)    // line 50
```

### Tile Types
```cpp
q_tile = st_bf<64, 128>   // CHUNK_SIZE x ATTN_F
v_tile = st_bf<64, 128>   // CHUNK_SIZE x ATTN_D
kv_state_tile = st_fl<128, 128>  // float32 state!
qk_map_tile = st_bf<128, 64>    // MLP featurization weight
```

### Key Pattern: Fused MLP featurization + sliding window + linear attention
This kernel combines:
1. MLP-based Q/K feature maps
2. Block-wise sliding window attention (within-chunk)
3. Linear attention via KV state propagation (across chunks)
4. Normalization across both components

### Softmax Feature Map (lines 39-46)
```cpp
warp::row_max(max_vec, tile);
warp::sub_row(tile, tile, max_vec);
warp::exp2(tile, tile);
warp::row_sum(sum_vec, tile);
warp::div_row(tile, tile, sum_vec);
```
This is a full row-wise softmax in registers using warp-level reductions.

---

## 13. Linear Attention (ALiBi-style)

**File**: `/home/kunwar/Work/kernel_libs/ThunderKittens/kernels/linear_attention/linear_attention.cu`

### Configuration
```cpp
NUM_WORKERS = 8, CHUNK_SIZE = 64, ATTN_D = 128, ATTN_F = 128
```

### ALiBi Decay Mask (lines 40-100)
The `wg_mask` function manually computes per-element exponential decays with `__expf(-1.0f * slope * (row - col))`. It uses magic bitmasks for diagonal elements:
```cpp
constexpr uint32_t MASK_X = 0xFF773311, MASK_Y = 0xF7733110;
```
These encode which thread-element pairs are on vs. above the diagonal in the register tile's distributed layout.

---

## 14. FFTConv

**File**: `/home/kunwar/Work/kernel_libs/ThunderKittens/kernels/fftconv/fftconv_pc.cu`

### Configuration (line 30)
```cpp
NUM_CONSUMER_WARPS=8, NUM_CONSUMER_WARPGROUPS=2
NUM_BLOCKS=1, OUTPUT_PIPE_STAGES=3, INPUT_PIPE_STAGES=3
```

### Tile Types
All 64x64 bf16 tiles, with complex variants (`cst_bf<64,64>`) for FFT operations.

### Memory Pattern
- Scratch block holds 7 complex 64x64 tiles (filter, FFT basis, etc.)
- Multiple batches packed into a single tile via `subtile<32,32>` calls
- 4 batch elements per warpgroup

---

## 15. Educational GEMM Progression (H100)

**File**: `/home/kunwar/Work/kernel_libs/ThunderKittens/kernels/gemm/educational_h100/level_08.cu`

### Final Optimized Configuration
```cpp
BLOCK_SIZE = 64           // tile dimension
M_BLOCK = 2               // consumer warpgroups along M
N_BLOCK = 4               // output tiles along N
NUM_PRODUCER_WORKERS = 4   // 1 warpgroup
NUM_CONSUMER_WORKERS = 8   // 2 warpgroups
// Total: 12 warps = 384 threads
```

### Register Budget
- Producer: `decrease_registers<40>()`
- Consumer: `increase_registers<232>()`
- Accumulator: `rt_fl<16, 64*4>` = `rt_fl<16, 256>` = 1x16 subtiles = **64 registers** for the wide accumulator

### Shared Memory
```cpp
st_bf<64,64> As[2][2]    // double-buffered, 2 M tiles
st_bf<64,64> Bs[2][4]    // double-buffered, 4 N tiles
st_bf<64,64> C_tiles[2][4]  // output staging
```

### TMA Usage
All loads/stores use TMA descriptors:
```cpp
tma::load_async(As[tic][m], g.A, {0, 0, row + m, 0}, bar);
tma::store_async(g.C, C_tiles[consumer_idx][n], {0, 0, row + consumer_idx, col + n});
```

---

## 16. Global Memory Load Pattern Details

**File**: `/home/kunwar/Work/kernel_libs/ThunderKittens/include/ops/group/memory/tile/global_to_shared.cuh`

Non-TMA loads use vectorized 128-bit (float4) loads:
```cpp
constexpr int elem_per_memcpy = sizeof(float4)/sizeof(typename ST::dtype);  // 8 for bf16
constexpr int memcpy_per_row = ST::cols / elem_per_memcpy;
constexpr int total_calls = (ST::rows*ST::cols + GROUP_THREADS*elem_per_memcpy-1) / (GROUP_THREADS*elem_per_memcpy);
```

Each thread loads one `float4` (16 bytes) per iteration, with threads striped across all tile elements. For a 64x64 bf16 tile (8192 bytes), 32 threads need `8192/16/32 = 16` iterations each.

---

## 17. Warp-Level Reduction Implementation

**File**: `/home/kunwar/Work/kernel_libs/ThunderKittens/include/ops/group/register/tile/reductions.cuh`

Row reduction in row-major register tiles (lines 21-60):
1. Accumulate across tile columns within each thread: `accum = op(src.tiles[i][j].data[k+0], src.tiles[i][j].data[k+1])` etc.
2. Pack x/y components: `accum_packed.x = op(accum_top.x, accum_top.y)`
3. **Warp shuffle**: `shfl_down_sync(delta=2)` then `shfl_down_sync(delta=1)`, then broadcast from leader thread via `shfl_sync(leader)`.

This uses the fact that in the NVIDIA MMA layout, threads with consecutive lane IDs hold adjacent column elements, so 2 shuffle steps suffice for a complete 8-wide reduction.

---

## Summary of Key Memory-Bound Patterns for IREE

| Kernel | Warps | Tiles | Pipe Stages | Shared Mem | Key Memory Pattern |
|--------|-------|-------|-------------|------------|-------------------|
| LayerNorm | 2 | sv_bf<1024> vectors | 2 (manual) | 25KB | Warp-per-row, double-buffer, full-row reduction |
| Rotary | 8C+4P | st_bf<16, headdim> | 3 in, 3 out | ~227KB | TMA, long-resident sin/cos in registers |
| Attention (H100) | 12C+4P | Q:64xD, KV:128xD | 2-4 | ~180KB | Producer-consumer pipeline, online softmax |
| Based LinAttn | 4 | 64x16 Q/K, 64x64 V | 2 (manual) | 98KB | Atomic cumsum, shuffle-based outer product |
| Mamba2 | 8C+4P | 64x64 all | 2 in, 2 out | ~227KB | Prefix sum decay in shared, persistent KV state |
| Flux GELU | BLOCK_M/16 C | 64xBLOCK_N output | default | ~227KB | Fused activation in registers post-GEMM |
| Hedgehog | 8 | 64x128 | 2 (manual) | ~227KB | Fused MLP + sliding window + linear attn |

Key takeaways for IREE configuration:
1. **Warp count scales with compute intensity**: Memory-bound ops use 2-4 warps; compute-bound ops use 12-16.
2. **Pipeline stages**: Memory-bound ops use 2 stages; compute-bound ops use 2-4.
3. **Register pressure is the binding constraint**: Consumer warps use 160-232 registers; producers use 32-40.
4. **Tile sizes are always multiples of 16** (the base MMA tile) and typically 64 or 128 in each dimension.
5. **Shared memory swizzle** is automatic based on tile column count and element size (32/64/128-byte swizzle).
6. **All reduction patterns** go through warp shuffles (shfl_down_sync by 2 then 1 for row-major) -- no cross-warp atomics except for special cases (Based linear attention).
7. **Memory-bound kernels** (layernorm, rotary) maximize data throughput by minimizing warps and maximizing memory-level parallelism via double-buffering and async loads.