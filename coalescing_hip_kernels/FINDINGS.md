# GPU Memory Coalescing Findings

Benchmarks run on AMD Radeon RX 9070 XT (RDNA4, gfx1201, 32 CUs, wave32, ~640 GB/s peak)
and AMD Instinct MI300X (CDNA3, gfx942, 304 CUs, wave64, ~5.3 TB/s peak).

Element type: f16. Vector widths: b16 (2B), b32 (4B), b64 (8B), b128 (16B).

## What is coalescing?

When threads in a wavefront issue memory requests, the hardware memory
controller merges requests to the same cache line into a single transaction.
Adjacent threads accessing adjacent memory addresses = all requests merge =
"coalesced." Scattered addresses = multiple cache line fetches = wasted
bandwidth.

## Key findings

### 1. Coalescing depends on the address SET, not the thread-to-address mapping

Reversed and XOR-swizzled lane orderings within a subgroup produce identical
bandwidth to normal ordering on both RDNA4 and CDNA3.

```
RDNA4 half8 loads:  normal=612, reversed=612, xor-half=612 GB/s
MI300X half8 loads: normal=4220, reversed=4195, xor-half=4193 GB/s
```

The hardware coalescer only cares that the wavefront's addresses fall within
the same cache lines. The permutation of which thread gets which address is
irrelevant.

### 2. Vector width matters for instruction throughput, not coalescing

With proper unrolling and enough in-flight loads, all vector widths (b16
through b128) achieve the same peak bandwidth for row-major access:

```
RDNA4 row-major loads:  b16=611, b32=612, b64=612, b128=612 GB/s
MI300X row-major loads: b16=3945, b32=3979, b64=4015, b128=3991 GB/s
```

Wider vectors reduce the number of load instructions needed, which matters
when the GPU is instruction-throughput limited (fewer CUs, less unrolling).
But with sufficient occupancy and unroll, the memory controller saturates
regardless of vector width.

### 3. Row-major vs column-major: cache locality matters, not coalescing

Both access patterns are coalesced (adjacent threads read adjacent elements).
The difference is the stride between consecutive accesses by the same thread:

- Row-major `mat[P][M][N]`: flat sequential access. Stride = N (small).
- Col-major `mat[M][P][N]`: stride between rows = P*N (large).

On RDNA4 (P=128, max stride = 128*256*2 = 64KB):

```
                    Row-major    Col-major
half8 loads:          612          613 GB/s   (no difference)
half1 loads:          611          613 GB/s   (no difference)
```

RDNA4's 64MB L2 cache absorbs the 64KB stride completely.

On MI300X (P=1216, max stride = 1216*512*2 = 1.2MB):

```
                    Row-major    Col-major
half8 loads:         3991         4220 GB/s   (col-major slightly faster)
half1 loads:         3945         4059 GB/s   (col-major slightly faster)
```

Even MI300X's 256MB L2 handles the 1.2MB stride. Col-major is slightly
faster because the subgroup-tiled pattern gives better memory-level
parallelism (different subgroups hit independent cache lines).

### 4. Store behavior differs from loads

Stores are fire-and-forget (no return data), so the write buffer absorbs
them differently:

- Row-major stores with narrow vectors (b16) suffer from cache line
  contention: multiple waves write to the same cache line, causing
  serialization.
- Col-major stores avoid this because each subgroup writes to independent
  cache lines (different rows, far apart).

```
RDNA4 stores:       Row-major    Col-major
  b16:                325          451 GB/s   (col-major 39% faster)
  b128:               600          611 GB/s   (similar)
```

As vector width increases, each wave fills more of a cache line, reducing
contention. At b128, each wave writes 512 bytes = 4 full cache lines with
no sharing.

### 5. Unroll factor is the dominant performance knob on large GPUs

MI300X with 304 CUs needs unroll=5-8 to generate enough outstanding memory
requests. RDNA4 with 32 CUs often saturates at unroll=1-4.

The unroll creates multiple independent loads/stores in the loop body,
allowing the hardware to pipeline memory requests without waiting for
each one to complete (avoids `s_waitcnt vmcnt(0)` serialization).

### 6. Triton's coalescing model is correct

Triton caps per-thread vector width at 128 bits (e.g., 4x f32 or 8x f16)
based on `min(alignment, contiguity, 128 / elemNumBits)`. Our benchmarks
confirm this is the right limit: going beyond 128 bits per thread (e.g.,
16x f16) would exceed the coalescing boundary and create gaps in the
wavefront's access pattern.

## Benchmark design

Input: `P x M x N` f16 elements where:
- P = 4 * num_cus (workgroups, one independent reduction each)
- N = subgroup_size * vec_elems (one coalesced load per thread per row)
- M = rows (large, constant M*N across vector widths)
- Total buffer ~2GB

Row-major layout `mat[P][M][N]`: each workgroup's slice is contiguous.
Kernel does flat grid-stride over the slice.

Col-major layout `mat[M][P][N]`: each workgroup's rows are interleaved
with other workgroups. Subgroups tile over rows, lanes over columns.
Stride between rows = P * N.

Loads reduce into float accumulators. Stores write `idx ^ threadIdx.x`
(cheap, unique per thread and iteration, prevents hardware compression).

Sweep: block_size in {subgroup*4, subgroup*8}, unroll in {1..8}, report
best. Cache flushed (512MB write) between iterations. Median of 10.
