# CUB/CCCL: Memory-Bound Kernel Analysis

---

# CUB Reduction / Scan / Transform: Exhaustive Configuration Analysis

## 1. TUNING POLICY ARCHITECTURE (The "Policy Hub" System)

CUB uses a **ChainedPolicy** pattern to select architecture-specific tuning at compile time. Each GPU architecture has a policy struct that inherits from the previous generation's policy. At runtime, the highest matching policy is used.

**File**: `/home/kunwar/Work/kernel_libs/cccl/cub/cub/device/dispatch/tuning/tuning_reduce.cuh`

The `policy_hub` defines policies per SM version:

| SM Version | Block Threads | Items/Thread (4B baseline) | Vec Load Length | Block Algorithm | Load Modifier | Measured BW |
|---|---|---|---|---|---|---|
| **SM 5.0** (GTX Titan) | 256 | 20 | 4 | BLOCK_REDUCE_WARP_REDUCTIONS | LOAD_LDG | 255.1 GB/s @ 48M 4B items |
| **SM 6.0** (P100) | 256 | 16 | 4 | BLOCK_REDUCE_WARP_REDUCTIONS | LOAD_LDG | 591 GB/s @ 64M 4B items |
| **SM 10.0** (Blackwell) | varies per type | varies | varies | BLOCK_REDUCE_WARP_REDUCTIONS | LOAD_LDG | tuned |

The new `policy_selector` (lines 345-398) replaces the legacy ChainedPolicy with a runtime-dispatchable function indexed by `arch_id`.

## 2. MemBoundScaling: THE KEY SCALING FORMULA

**File**: `/home/kunwar/Work/kernel_libs/cccl/cub/cub/util_arch.cuh`, lines 150-170

This is the core formula CUB uses to adapt tuning constants to different data type sizes:

```cpp
constexpr auto scale_mem_bound(int nominal_4B_block_threads,
                                int nominal_4B_items_per_thread,
                                int target_type_size) -> scaling_result
{
  const int items_per_thread =
    clamp(nominal_4B_items_per_thread * 4 / target_type_size,
          1,
          nominal_4B_items_per_thread * 2);
  const int block_threads =
    min(nominal_4B_block_threads,
        round_up(max_smem_per_block / (target_type_size * items_per_thread), 32));
  return {items_per_thread, block_threads};
}
```

**How it works**:
- All tuning constants are specified as "nominal 4-byte" values
- `items_per_thread` is scaled inversely proportional to type size: `nominal * 4 / sizeof(T)`, clamped to `[1, 2 * nominal]`
- `block_threads` is capped by shared memory: `48KB / (sizeof(T) * items_per_thread)`, rounded up to a multiple of 32
- For a 1-byte type with nominal 16 IPT: `16 * 4 / 1 = 64` but clamped to `2 * 16 = 32`
- For an 8-byte type with nominal 16 IPT: `16 * 4 / 8 = 8`
- For a 16-byte type: `16 * 4 / 16 = 4`

There is also `NoScaling` (line 172-177) which passes through the nominal values unchanged -- used in the kernel instantiation when the policy selector has already done the scaling.

There is also `RegBoundScaling` (line 129-148) for register-bound kernels, which uses `max(1, nominal * 4 / max(4, sizeof(T)))` -- note the `max(4, ...)` means types <= 4 bytes are NOT scaled.

## 3. SM100 (Blackwell) REDUCE TUNING CONSTANTS

**File**: `/home/kunwar/Work/kernel_libs/cccl/cub/cub/device/dispatch/tuning/tuning_reduce.cuh`, lines 169-245

These were empirically tuned:

| Data Type | Offset Size | Accum Size | Items/Thread | Block Threads | Vec Load Length |
|---|---|---|---|---|---|
| `float` (sum) | 4 | 4 | 16 | 512 | 2 |
| `double` (sum) | 4 | 8 | 16 | 640 | 1 |
| generic 8B (sum) | 4 | 8 | 15 | 512 | 2 |
| generic 8B (sum) | 8 | 8 | 15 | 512 | 1 |

**Critical observation**: Min/max operations showed no significant improvement over the base policy, so they have no SM100 specializations (fall through to SM 6.0 defaults).

After these nominal values, `scale_mem_bound` is still applied on top (line 359), which means these are already the tuned values for those specific type sizes, then `scale_mem_bound` may adjust further. Wait -- looking more carefully at the `policy_selector::operator()` (line 352-397), the SM100 tuning values ARE passed through `scale_mem_bound`:

```cpp
auto [scaled_items, scaled_threads] = scale_mem_bound(sm100_tuning->threads, sm100_tuning->items, accum_size);
```

So for `float` (accum_size=4): `scale_mem_bound(512, 16, 4)` => `items = clamp(16*4/4, 1, 32) = 16`, `threads = min(512, ...) = 512`. The scaling is a no-op for 4B since the tuning was done for 4B.

For `double` (accum_size=8): `scale_mem_bound(640, 16, 8)` => `items = clamp(16*4/8, 1, 32) = 8`, `threads = min(640, ...)`. So the effective config for double on SM100 is **640 threads, 8 items/thread**.

## 4. VECTORIZED LOADING IN REDUCTION

**File**: `/home/kunwar/Work/kernel_libs/cccl/cub/cub/agent/agent_reduce.cuh`, lines 199-343

### The Vectorization Decision (lines 225-229)
```cpp
static constexpr bool ATTEMPT_VECTORIZATION =
  (VECTOR_LOAD_LENGTH > 1) && (ITEMS_PER_THREAD % VECTOR_LOAD_LENGTH == 0)
  && (is_pointer<InputIteratorT>)
  && (is_primitive<InputT> || is_trivially_relocatable<InputT>);
```

Requirements for vectorization:
1. `VECTOR_LOAD_LENGTH > 1` (policy must request it)
2. `ITEMS_PER_THREAD` must be divisible by `VECTOR_LOAD_LENGTH`
3. Input must be a raw pointer (not a fancy iterator)
4. Input type must be primitive or trivially relocatable

### Runtime Alignment Check (lines 258-270)
Even when compile-time vectorization is possible, a runtime alignment check is performed:
```cpp
static bool IsAligned(Iterator d_in) {
  return is_sufficiently_aligned<alignof(VectorT)>(d_in);
}
```

### Vectorized Load Path (ConsumeFullTile, lines 302-343)
When vectorizing:
1. Cast the input pointer to a `VectorT*` (e.g., `float4*` for `float` with vec length 4)
2. Load in **striped** fashion: thread `i` loads `d_vec_in[NumThreads * j]` for `j = 0..words-1`
3. This gives coalesced access: consecutive threads load consecutive vector elements
4. Each vector element is a multi-element structure (e.g., `float4` = 4 floats)

The effective memory transaction per thread:
- `VECTOR_LOAD_LENGTH` elements per vector load
- `ITEMS_PER_THREAD / VECTOR_LOAD_LENGTH` vector loads per tile
- Total: `ITEMS_PER_THREAD` elements per thread per tile

**Example**: float, SM 6.0: 256 threads, 16 items/thread, vec_load=4
- `VectorT = float4` (16 bytes)
- 4 vector loads per thread, each loading 4 floats
- Tile size = 256 * 16 = 4096 elements = 16KB per tile
- Each thread does 4 loads of 16 bytes = 64 bytes total

### Scalar Path (lines 337-343)
When not vectorizing (fancy iterator, misaligned pointer):
- Striped load: thread `i` loads `d_in[block_offset + i]`, `d_in[block_offset + i + NumThreads]`, etc.
- This is still coalesced for consecutive threads but uses scalar loads

## 5. REDUCTION DECOMPOSITION: Thread -> Warp -> Block

### Thread-Level: `ThreadReduce` 
**File**: `/home/kunwar/Work/kernel_libs/cccl/cub/cub/thread/thread_reduce.cuh`

Each thread reduces its `ITEMS_PER_THREAD` elements using a sequential loop. Optimizations:
- SIMD/DPX instructions for int16/bf16/half on SM70+/SM80+/SM90+
- Ternary tree reduction for ILP on min/max operations (SM50+ for int, SM80+ for half)

### Warp-Level: `WarpReduceShfl`
**File**: `/home/kunwar/Work/kernel_libs/cccl/cub/cub/warp/specializations/warp_reduce_shfl.cuh`

Used when `LOGICAL_WARP_THREADS` is a power of two (always true for 32-thread warps).

**The shuffle tree** (lines 457-478):
```cpp
template <int STEP>
void ReduceStep(T& input, ReductionOp op, int last_lane, constant_t<STEP>) {
  input = ReduceStep(input, op, last_lane, 1 << STEP);  // offset = 1, 2, 4, 8, 16
  ReduceStep(input, op, last_lane, constant_v<STEP + 1>);  // recurse
}
```

This is a **Kogge-Stone** parallel reduction using `shfl.sync.down`:
- Step 0: offset=1 (pairs of adjacent lanes)
- Step 1: offset=2
- Step 2: offset=4
- Step 3: offset=8
- Step 4: offset=16
- Total: 5 steps for 32-thread warp = `log2(32)` steps

**Specialized ReduceStep for common types** (lines 163-336):
- `unsigned int + plus`: Fused `shfl.sync.down + add.u32` in single PTX asm block
- `float + plus`: Fused `shfl.sync.down + add.f32`
- `unsigned long long + plus`: Two `shfl.sync.down` (lo/hi halves) + `add.u64`
- `double + plus`: Two `shfl.sync.down` (lo/hi) + `add.f64`
- `long long + plus`: Two `shfl.sync.down` + `add.s64`

**SM80+ hardware reduction** (lines 502-508):
For integral types <= 4 bytes with plus/min/max/bitwise ops, CUB uses `__reduce_add_sync`, `__reduce_min_sync`, etc. -- single-instruction warp-wide reductions.

### Block-Level: `BlockReduceWarpReductions`
**File**: `/home/kunwar/Work/kernel_libs/cccl/cub/cub/block/specializations/block_reduce_warp_reductions.cuh`

This is the algorithm used by ALL reduce policies (`BLOCK_REDUCE_WARP_REDUCTIONS`).

Structure:
1. **Warp reduce**: Each warp does an independent `WarpReduceShfl::Reduce`
2. **Cross-warp combine** (deterministic): Lane 0 of each warp writes its aggregate to `shared_mem[warp_id]`. Thread 0 then sequentially reduces all warp aggregates (line 158-167):
   ```cpp
   for (int warp_idx = 1; warp_idx < warps; ++warp_idx) {
     T addend = temp_storage.warp_aggregates[warp_idx];
     warp_aggregate = reduction_op(warp_aggregate, addend);
   }
   ```
3. **Cross-warp combine** (nondeterministic): Warp 0's lane 0 stores, other warps use `atomicAdd` to accumulate into a single shared memory location.

**Number of warps**: `ceil_div(block_threads, 32)`. For 256 threads = 8 warps, 512 threads = 16 warps.

Only **one** `__syncthreads()` barrier is needed.

## 6. MULTI-BLOCK REDUCTION: Two-Pass Strategy

**File**: `/home/kunwar/Work/kernel_libs/cccl/cub/cub/device/dispatch/dispatch_reduce.cuh`, lines 330-454

### Decision Point (lines 440-454)
```cpp
if (num_items <= block_threads * items_per_thread) {
  // Single tile: one block handles everything
  InvokeSingleTile(kernel_source.SingleTileKernel(), ...);
} else {
  // Multi-block: two-pass
  InvokePasses(kernel_source.ReductionKernel(), 
               kernel_source.SingleTileSecondKernel(), ...);
}
```

For SM 6.0, 4-byte data: threshold = 256 * 16 = 4096 elements.

### Multi-Block Grid Sizing (lines 334-353)
```cpp
int sm_count = ...; // number of SMs on device
KernelConfig reduce_config;
reduce_config.Init(reduce_kernel, active_policy.Reduce(), launcher_factory);

int reduce_device_occupancy = reduce_config.sm_occupancy * sm_count;
int max_blocks = reduce_device_occupancy * subscription_factor;  // subscription_factor = 5
```

**Occupancy**: CUB queries `cudaOccupancyMaxActiveBlocksPerMultiprocessor` via `launcher_factory.MaxSmOccupancy`. This gives the number of concurrent thread blocks per SM.

**Grid size**: `occupancy_per_SM * num_SMs * 5` (subscription factor of 5x oversubscription).

### Work Distribution: GridEvenShare with STRIP_MINE
**File**: `/home/kunwar/Work/kernel_libs/cccl/cub/cub/grid/grid_even_share.cuh`

The multi-block reduce uses `GRID_MAPPING_STRIP_MINE` (line 433 in agent_reduce.cuh):
```cpp
// block_stride = grid_size * TILE_ITEMS
// block_offset = blockIdx.x * TILE_ITEMS
// block_end = num_items
```

Each block starts at `blockIdx.x * TILE_ITEMS` and strides by `grid_size * TILE_ITEMS`. This means:
- Block 0 processes tiles 0, grid_size, 2*grid_size, ...
- Block 1 processes tiles 1, grid_size+1, ...
- This interleaves blocks across the data for better L2 cache utilization

### Pass 2: Second Kernel
After pass 1 produces `grid_size` partial reductions (one per block), pass 2 launches a **single block** to reduce them using `DeviceReduceSingleTileKernel`.

Temporary storage = `grid_size * sizeof(AccumT)`.

## 7. SCAN TUNING CONSTANTS

**File**: `/home/kunwar/Work/kernel_libs/cccl/cub/cub/device/dispatch/tuning/tuning_scan.cuh`

Scan has more parameters than reduce because it must also store results:

### SM 9.0 (Hopper) Tunings
| Accum Size | Block Threads | Items/Thread | Load Algorithm | Store Algorithm |
|---|---|---|---|---|
| 1B | 192 | 22 | WARP_TRANSPOSE | WARP_TRANSPOSE |
| 2B | 512 | 12 | WARP_TRANSPOSE | WARP_TRANSPOSE |
| 4B (generic) | 128 | 24 | WARP_TRANSPOSE | WARP_TRANSPOSE |
| 4B (float) | 128 | 24 | WARP_TRANSPOSE | WARP_TRANSPOSE |
| 8B (generic) | 224 | 24 | WARP_TRANSPOSE | WARP_TRANSPOSE |
| 8B (double) | 224 | 24 | WARP_TRANSPOSE | WARP_TRANSPOSE |
| 16B (int128) | 576 | 21 | WARP_TRANSPOSE | WARP_TRANSPOSE |

### SM 10.0 (Blackwell) Scan Tunings - Sum, Offset=4
| Value Size | Block Threads | Items/Thread | Delay Constructor |
|---|---|---|---|
| 1B | 512 | 18 | exponential_backon(768, 820) |
| 2B | 512 | 13 | exponential_backon(1384, 720) |
| 4B | 384 | 22 | exponential_backon_jitter(1904, 830) |
| 8B | 416 | 23 | exponential_backon_jitter_window(772, 710) |

The scan also introduces a `delay_constructor` for controlling the spin-wait delay in the decoupled look-back algorithm. These are empirically tuned per architecture.

### Default Scan Policy (SM 6.0+)
```
block_threads = 128, items_per_thread = 15
load = BLOCK_LOAD_WARP_TRANSPOSE, store = BLOCK_STORE_WARP_TRANSPOSE
scan_algorithm = BLOCK_SCAN_WARP_SCANS
```

The scan uses `MemBoundScaling` in its `select_agent_policy` (lines 559), so these nominal 4B values are scaled the same way as reduce.

## 8. TRANSFORM (ELEMENTWISE) CONFIGURATION

**File**: `/home/kunwar/Work/kernel_libs/cccl/cub/cub/device/dispatch/tuning/tuning_transform.cuh`

Transform uses a fundamentally different approach -- runtime items-per-thread selection:

### Three Algorithms:
1. **Prefetch**: Simple grid-stride loop with explicit L2 prefetch instructions
2. **Vectorized**: Uses vector loads/stores (e.g., `int4`)  
3. **Async copy** (SM 9.0+): Uses `cp.async.bulk` / TMA for global -> shared memory

### Prefetch Policy Defaults:
```cpp
prefetch_policy{
  .block_threads = 256,
  .min_items_per_thread = 1,
  .max_items_per_thread = 32,
  .prefetch_byte_stride = 128  // one cache line
}
```

### Vectorized Policy (per architecture):
| Arch | Block Threads | Items/Thread | Vec Size | Notes |
|---|---|---|---|---|
| SM 8.0 (A100) | 128 | 16 | 4 | triad kernel |
| SM 8.0 fill | 256 | 8 | max(8/sizeof(T), 1) | 64-bit instructions |
| SM 9.0+ fill, type <= 4B | 256 | 16 | max(8/sizeof(T), 1) | |
| SM 9.0+ fill, type > 4B | 128 | 16 | max(8/sizeof(T), 1) | |
| Default | 256 | 8 | 4 | |

### Bytes-in-Flight Targets (line 271-286):
```cpp
SM 10.0+ (B200):  64 * 1024 = 65536 bytes
SM 9.0+  (H100):  48 * 1024 = 49152 bytes
SM 8.0+  (A100):  16 * 1024 = 16384 bytes
< SM 8.0 (V100):  12 * 1024 = 12288 bytes
```

These represent the minimum amount of data that should be "in flight" per SM to saturate the memory subsystem.

### Runtime Items/Thread Selection:
Transform dynamically adjusts `items_per_thread`:
```cpp
items_per_thread = ceil_div(min_bytes_in_flight, 
                            max_occupancy * block_threads * loaded_bytes_per_iter);
```
Then spread out to fill occupancy:
```cpp
items_per_thread = min(items_per_thread, 
                       ceil_div(num_items, sm_count * block_threads * max_occupancy));
```

### Async Copy (SM 9.0+):
```cpp
SM 9.0: block_threads = 256, bulk_copy_alignment = 128
SM 10.0: block_threads = 128, bulk_copy_alignment = 16
```

## 9. OCCUPANCY REASONING

CUB does **not** hard-code occupancy. Instead:

1. **Query**: `cudaOccupancyMaxActiveBlocksPerMultiprocessor(kernel, block_threads, dyn_smem)` gives actual occupancy
2. **Oversubscribe**: Grid size = `occupancy * SM_count * 5` (the subscription factor of 5)
3. **The trade-off**: More items/thread means fewer blocks needed but lower occupancy. CUB's tuning resolves this empirically -- the tuned constants already balance occupancy vs. work-per-thread.

For reduce, occupancy is implicitly managed by the shared memory constraint in `scale_mem_bound`:
```
block_threads = min(nominal, round_up(48KB / (sizeof(T) * items_per_thread), 32))
```

For transform, occupancy is explicitly measured at runtime and used to compute items_per_thread.

## 10. MEMORY ACCESS PATTERN SUMMARY

### Reduce:
- **Full tiles**: Striped access with optional vectorization
  - Thread `t` loads elements at offsets `t`, `t + NumThreads`, `t + 2*NumThreads`, ...
  - With vectorization: loads `VectorT` at offsets scaled by vector width
  - Coalesced: consecutive threads access consecutive memory addresses within each "stripe"
- **Partial tiles**: Scalar striped access with bounds checking via while loop
- **Multi-block**: Strip-mine mapping (interleaved blocks) for L2 locality

### Scan:
- **BLOCK_LOAD_WARP_TRANSPOSE**: Threads load in striped order, then exchange via shared memory to get blocked order
- **BLOCK_STORE_WARP_TRANSPOSE**: Reverse of load -- blocked to striped via shared memory exchange
- This ensures coalesced global memory access while maintaining blocked thread-local data for scan

### Transform:
- **Prefetch**: Explicit prefetch instructions for 128-byte cache lines, striped access
- **Vectorized**: Vector loads/stores (up to 16 bytes per transaction)
- **Async copy**: DMA from global to shared memory, then compute from shared

## 11. KEY NUMERICAL REFERENCE TABLE

**Effective tile sizes for SM 6.0+ reduce (after MemBoundScaling)**:

| sizeof(T) | Nominal IPT=16, BT=256 | Scaled IPT | Scaled BT | Tile Size (elements) | Tile Size (bytes) |
|---|---|---|---|---|---|
| 1 | 16 * 4 / 1 = 64 -> clamped to 32 | 32 | 256 | 8192 | 8192 |
| 2 | 16 * 4 / 2 = 32 -> clamped to 32 | 32 | 256 | 8192 | 16384 |
| 4 | 16 * 4 / 4 = 16 | 16 | 256 | 4096 | 16384 |
| 8 | 16 * 4 / 8 = 8 | 8 | 256 | 2048 | 16384 |
| 16 | 16 * 4 / 16 = 4 | 4 | 256 | 1024 | 16384 |

Notice: the tile size in bytes converges to ~16KB for types >= 4 bytes. The MemBoundScaling keeps the total bytes-per-tile roughly constant, which is exactly what you want for a memory-bound kernel.

---

**Key files referenced**:
- `/home/kunwar/Work/kernel_libs/cccl/cub/cub/util_arch.cuh` -- MemBoundScaling, scale_mem_bound, subscription_factor
- `/home/kunwar/Work/kernel_libs/cccl/cub/cub/device/dispatch/tuning/tuning_reduce.cuh` -- All reduce tuning policies and SM100 constants
- `/home/kunwar/Work/kernel_libs/cccl/cub/cub/device/dispatch/tuning/tuning_scan.cuh` -- All scan tuning policies
- `/home/kunwar/Work/kernel_libs/cccl/cub/cub/device/dispatch/tuning/tuning_transform.cuh` -- Transform policies including bytes-in-flight
- `/home/kunwar/Work/kernel_libs/cccl/cub/cub/agent/agent_reduce.cuh` -- Core reduction agent with vectorized load logic
- `/home/kunwar/Work/kernel_libs/cccl/cub/cub/device/dispatch/kernels/kernel_reduce.cuh` -- Kernel entry points
- `/home/kunwar/Work/kernel_libs/cccl/cub/cub/device/dispatch/dispatch_reduce.cuh` -- Two-pass orchestration and grid sizing
- `/home/kunwar/Work/kernel_libs/cccl/cub/cub/warp/specializations/warp_reduce_shfl.cuh` -- Warp shuffle reduction with PTX asm
- `/home/kunwar/Work/kernel_libs/cccl/cub/cub/block/specializations/block_reduce_warp_reductions.cuh` -- Block-level warp aggregation
- `/home/kunwar/Work/kernel_libs/cccl/cub/cub/grid/grid_even_share.cuh` -- Work distribution across blocks
- `/home/kunwar/Work/kernel_libs/cccl/cub/cub/device/dispatch/tuning/common.cuh` -- Type/op classification enums