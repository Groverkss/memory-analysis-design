=== Coalescing Benchmark Results ===
Date:       2026-03-26 10:58:37 UTC
Host:       tiny
GPU:        AMD Radeon RX 9070 XT (gfx1201)
ROCm:       6.4.2-120
hipcc:      HIP version: 6.4.43484-123eb5128
Config:     50 iterations per kernel, median reported, cache flushed between iterations


Element type: f32 (4 bytes)
Buffer: 256 MB (64M floats)
Block size: 256 threads, Iterations: 50 (median reported)

Kernel                                       Median        Min        Max
---------------------------------------- ---------- ---------- ----------

  == COALESCED LOADS (contiguous f32s per thread) ==
  1x f32 = 1*4B contiguous                  284.9    283.1    285.3
  2x f32 = 2*4B contiguous                  347.0    343.5    353.2
  4x f32 = 4*4B contiguous                  404.7    404.2    410.0
  8x f32 = 8*4B contiguous                  440.1    436.0    440.5
  16x f32 = 16*4B contiguous                454.5    453.7    462.0

  == GAPPED LOADS (strided f32s per thread) ==
  1x f32 stride=2 (1*4B/thread)             173.5    172.2    175.5
  1x f32 stride=4 (1*4B/thread)             101.2    101.1    102.5
  1x f32 stride=32 (1*4B/thread)             14.3     14.3     14.4
  2x f32 stride=2 (2*4B/thread)             202.3    200.7    202.6
  2x f32 stride=4 (2*4B/thread)             110.0    109.3    111.7
  2x f32 stride=32 (2*4B/thread)             14.5     14.5     14.8
  4x f32 stride=2 (4*4B/thread)             220.0    219.2    220.3
  4x f32 stride=4 (4*4B/thread)             113.6    112.6    115.2
  4x f32 stride=32 (4*4B/thread)             14.6     14.4     14.8

  == COALESCED STORES (contiguous f32s per thread) ==
  1x f32 = 1*4B contiguous                  619.1    611.5    619.7
  2x f32 = 2*4B contiguous                  624.6    623.7    625.1
  4x f32 = 4*4B contiguous                  627.1    619.5    627.7
  8x f32 = 8*4B contiguous                  627.4    626.4    628.4
  16x f32 = 16*4B contiguous                626.6    620.4    628.0

  == GAPPED STORES (strided f32s per thread) ==
  1x f32 stride=2 (1*4B/thread)             125.7    125.0    126.8
  1x f32 stride=4 (1*4B/thread)              62.9     62.6     63.1
  1x f32 stride=32 (1*4B/thread)             46.1     43.7     47.5
  2x f32 stride=2 (2*4B/thread)             125.7    125.0    126.0
  2x f32 stride=4 (2*4B/thread)              62.5     62.1     62.7
  2x f32 stride=32 (2*4B/thread)             39.1     38.2     40.8
  4x f32 stride=2 (4*4B/thread)             124.9    124.4    125.2
  4x f32 stride=4 (4*4B/thread)              62.3     62.0     62.7
  4x f32 stride=32 (4*4B/thread)             36.4     36.0     36.9

