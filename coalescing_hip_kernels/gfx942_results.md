=== Coalescing Benchmark Results ===
Date:       2026-03-26 10:58:57 UTC
Host:       sharkmi300x-2
GPU:        AMD Instinct MI300X (gfx942)
ROCm:       7.2.0
hipcc:      HIP version: 7.2.26015-fc0010cf6a
Config:     50 iterations per kernel, median reported, cache flushed between iterations


Element type: f32 (4 bytes)
Buffer: 256 MB (64M floats)
Block size: 256 threads, Iterations: 50 (median reported)

Kernel                                       Median        Min        Max
---------------------------------------- ---------- ---------- ----------

  == COALESCED LOADS (contiguous f32s per thread) ==
  1x f32 = 1*4B contiguous                 1737.3   1281.0   1767.1
  2x f32 = 2*4B contiguous                 2137.4   2055.5   2195.6
  4x f32 = 4*4B contiguous                 2456.4   2374.5   2524.9
  8x f32 = 8*4B contiguous                 2578.3   2505.0   2738.4
  16x f32 = 16*4B contiguous               2663.3   2594.2   2729.5

  == GAPPED LOADS (strided f32s per thread) ==
  1x f32 stride=2 (1*4B/thread)            1081.5   1034.1   1116.8
  1x f32 stride=4 (1*4B/thread)             618.6    593.2    629.1
  1x f32 stride=32 (1*4B/thread)             84.4     80.0     86.9
  2x f32 stride=2 (2*4B/thread)            1239.1   1182.2   1269.6
  2x f32 stride=4 (2*4B/thread)             637.7    619.8    683.8
  2x f32 stride=32 (2*4B/thread)             80.0     77.6     84.4
  4x f32 stride=2 (4*4B/thread)            1282.2   1248.3   1328.5
  4x f32 stride=4 (4*4B/thread)             660.6    635.0    683.8
  4x f32 stride=32 (4*4B/thread)             80.9     75.6     82.7

  == COALESCED STORES (contiguous f32s per thread) ==
  1x f32 = 1*4B contiguous                 2045.5   1983.7   2051.1
  2x f32 = 2*4B contiguous                 3822.6   3714.5   3837.9
  4x f32 = 4*4B contiguous                 4233.7   4063.0   4473.9
  8x f32 = 8*4B contiguous                 3187.9   3069.6   3229.4
  16x f32 = 16*4B contiguous               2081.0   2030.6   2090.1

  == GAPPED STORES (strided f32s per thread) ==
  1x f32 stride=2 (1*4B/thread)             871.8    847.8    886.1
  1x f32 stride=4 (1*4B/thread)             443.8    433.3    455.7
  1x f32 stride=32 (1*4B/thread)             88.1     83.7     92.5
  2x f32 stride=2 (2*4B/thread)             879.4    841.4    905.5
  2x f32 stride=4 (2*4B/thread)             449.7    438.1    459.6
  2x f32 stride=32 (2*4B/thread)             84.2     64.0     87.8
  4x f32 stride=2 (4*4B/thread)             864.4    813.0    899.5
  4x f32 stride=4 (4*4B/thread)             432.3    408.6    452.6
  4x f32 stride=32 (4*4B/thread)             79.1     76.5     81.1

