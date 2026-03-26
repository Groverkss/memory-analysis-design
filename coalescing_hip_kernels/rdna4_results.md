# RDNA4 Coalescing Benchmark Results

**GPU:** AMD Radeon RX 9070 XT (gfx1201)
**Theoretical peak bandwidth:** ~608 GB/s
**Buffer size:** 256 MB (64M floats)
**Wave size:** wave32
**Block size:** 256 threads
**Date:** 2026-03-26

## Coalesced Loads

| EPT | GB/s | % of peak | ISA instruction |
|-----|------|-----------|-----------------|
| 1   | 304  | 50%       | `global_load_b32` × 1 |
| 2   | 404  | 66%       | `global_load_b64` × 1 |
| 4   | 487  | 80%       | `global_load_b128` × 1 |
| 8   | 551  | 91%       | `global_load_b128` × 2 |
| 16  | 578  | 95%       | `global_load_b128` × 4 |

Wider vector loads significantly improve bandwidth. The compiler vectorizes
EPT=2 into b64, EPT=4+ into b128 instructions. Each load requires a
`s_wait_loadcnt` before the data can be used, so fewer wider loads = less
latency overhead.

## Gapped Loads

| EPT | Gap | GB/s | vs coalesced EPT=1 |
|-----|-----|------|--------------------|
| 1   | 2   | 202  | 0.66x |
| 1   | 4   | 122  | 0.40x |
| 1   | 32  | 19   | 0.06x |
| 2   | 2   | 243  | 0.80x |
| 2   | 4   | 138  | 0.45x |
| 2   | 32  | 19   | 0.06x |
| 4   | 2   | 276  | 0.91x |
| 4   | 4   | 145  | 0.48x |
| 4   | 32  | 19   | 0.06x |

Bandwidth scales roughly as 1/gap. Gap=32 (stride across a full wave) drops
to ~19 GB/s regardless of EPT. Increasing EPT helps slightly at small gaps
but cannot recover the fundamental stride penalty.

## Coalesced Stores

| EPT | GB/s | % of peak | ISA instruction |
|-----|------|-----------|-----------------|
| 1   | 637  | 105%*     | `global_store_b32` × 1 |
| 2   | 637  | 105%*     | `global_store_b64` × 1 |
| 4   | 637  | 105%*     | `global_store_b128` × 1 |
| 8   | 637  | 105%*     | `global_store_b128` × 2 |
| 16  | 635  | 104%*     | `global_store_b128` × 4 |

*Over 100% because the theoretical peak is approximate.

EPT has no effect on store bandwidth. The ISA confirms the compiler does
emit different-width stores (b32, b64, b128), but stores are fire-and-forget:
the thread pushes writes into a write buffer without waiting. The memory
controller coalesces the wave's writes into full cache line transactions
regardless of per-thread vector width.

## Gapped Stores

| EPT | Gap | GB/s | vs coalesced EPT=1 |
|-----|-----|------|--------------------|
| 1   | 2   | 159  | 0.25x |
| 1   | 4   | 79   | 0.12x |
| 1   | 32  | 53   | 0.08x |
| 2   | 2   | 158  | 0.25x |
| 2   | 4   | 79   | 0.12x |
| 2   | 32  | 46   | 0.07x |
| 4   | 2   | 157  | 0.25x |
| 4   | 4   | 79   | 0.12x |
| 4   | 32  | 43   | 0.07x |

Gapped stores scale as ~1/gap. EPT has no effect within a gap level (same
fire-and-forget behavior). Gapped stores are worse than gapped loads at
gap=32 (43 vs 19 GB/s in absolute terms, but loads read less data total
in the gapped case).

## Key Takeaways

1. **For loads, vector width matters.** EPT=1 achieves 50% of peak; EPT=16
   reaches 95%. The load latency pipeline benefits from fewer, wider
   instructions.

2. **For stores, vector width is irrelevant.** The write buffer and memory
   controller coalesce wave-level writes regardless of per-thread store width.
   All EPT values hit ~637 GB/s.

3. **Gaps are devastating.** A stride of 32 (one wave width) drops load
   bandwidth to 3% of peak and store bandwidth to 7%. Each cache line is
   fetched/written but only one element is used.

4. **Loads vs stores asymmetry.** Pure stores exceed pure loads at all EPT
   levels because stores don't block on memory latency. The load kernel also
   includes a reduce (accumulate) + one write-back, adding overhead.
