// Coalescing benchmark
//
// Tests load/store bandwidth with varying vector widths and gap patterns.
// For each kernel type, sweeps block sizes (subgroup*4, subgroup*8) and
// unroll factors (1-8), reports the best configuration.
//
// Load kernels: grid-stride loop, batch reduce into register accumulators.
// Store kernels: grid-stride loop, write iota (dst[i] = i).
//
// Vector widths (128-bit max = coalescing limit for f32):
//   float   = 1x f32 =  4B  -> global_load_b32
//   float2  = 2x f32 =  8B  -> global_load_b64
//   float4  = 4x f32 = 16B  -> global_load_b128

#include <hip/hip_runtime.h>
#include <algorithm>
#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define HIP_CHECK(x)                                                           \
  do {                                                                         \
    hipError_t err = (x);                                                      \
    if (err != hipSuccess) {                                                   \
      fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(err),      \
              __FILE__, __LINE__);                                             \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// ---------------------------------------------------------------------------
// Cache flush
// ---------------------------------------------------------------------------
static float *d_flush = nullptr;
static const size_t FLUSH_BYTES = 512ULL * 1024 * 1024;

__global__ void flush_kernel(float *buf, int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n)
    buf[tid] = (float)tid;
}

static void flush_cache() {
  int n = FLUSH_BYTES / sizeof(float);
  int block = 256;
  int grid = (n + block - 1) / block;
  flush_kernel<<<grid, block>>>(d_flush, n);
  HIP_CHECK(hipDeviceSynchronize());
}

// ---------------------------------------------------------------------------
// Coalesced load kernels — templated on UNROLL.
// Grid-stride loop with UNROLL independent accumulators.
// ---------------------------------------------------------------------------

template <int UNROLL>
__global__ void coalesced_load_f32(const float *__restrict__ src,
                                   float *__restrict__ dst, int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  float acc[UNROLL];
  for (int u = 0; u < UNROLL; ++u) acc[u] = 0.0f;
  int i = tid;
  for (; i + (UNROLL - 1) * stride < n; i += UNROLL * stride) {
    #pragma unroll
    for (int u = 0; u < UNROLL; ++u)
      acc[u] += src[i + u * stride];
  }
  for (; i < n; i += stride)
    acc[0] += src[i];
  float total = 0.0f;
  for (int u = 0; u < UNROLL; ++u) total += acc[u];
  dst[tid] = total;
}

template <int UNROLL>
__global__ void coalesced_load_f32x2(const float2 *__restrict__ src,
                                     float *__restrict__ dst, int n2) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  float acc[UNROLL];
  for (int u = 0; u < UNROLL; ++u) acc[u] = 0.0f;
  int i = tid;
  for (; i + (UNROLL - 1) * stride < n2; i += UNROLL * stride) {
    #pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      float2 v = src[i + u * stride];
      acc[u] += v.x + v.y;
    }
  }
  for (; i < n2; i += stride) {
    float2 v = src[i];
    acc[0] += v.x + v.y;
  }
  float total = 0.0f;
  for (int u = 0; u < UNROLL; ++u) total += acc[u];
  dst[tid] = total;
}

template <int UNROLL>
__global__ void coalesced_load_f32x4(const float4 *__restrict__ src,
                                     float *__restrict__ dst, int n4) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  float acc[UNROLL];
  for (int u = 0; u < UNROLL; ++u) acc[u] = 0.0f;
  int i = tid;
  for (; i + (UNROLL - 1) * stride < n4; i += UNROLL * stride) {
    #pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      float4 v = src[i + u * stride];
      acc[u] += v.x + v.y + v.z + v.w;
    }
  }
  for (; i < n4; i += stride) {
    float4 v = src[i];
    acc[0] += v.x + v.y + v.z + v.w;
  }
  float total = 0.0f;
  for (int u = 0; u < UNROLL; ++u) total += acc[u];
  dst[tid] = total;
}

// ---------------------------------------------------------------------------
// Gapped load kernels
// ---------------------------------------------------------------------------

template <int UNROLL, int GAP>
__global__ void gapped_load_f32(const float *__restrict__ src,
                                float *__restrict__ dst, int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int step = gridDim.x * blockDim.x * GAP;
  float acc[UNROLL];
  for (int u = 0; u < UNROLL; ++u) acc[u] = 0.0f;
  int i = tid * GAP;
  for (; i + (UNROLL - 1) * step < n; i += UNROLL * step) {
    #pragma unroll
    for (int u = 0; u < UNROLL; ++u)
      acc[u] += src[i + u * step];
  }
  for (; i < n; i += step)
    acc[0] += src[i];
  float total = 0.0f;
  for (int u = 0; u < UNROLL; ++u) total += acc[u];
  dst[tid] = total;
}

template <int UNROLL, int GAP>
__global__ void gapped_load_f32x4(const float4 *__restrict__ src,
                                  float *__restrict__ dst, int n4) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int step = gridDim.x * blockDim.x * GAP;
  float acc[UNROLL];
  for (int u = 0; u < UNROLL; ++u) acc[u] = 0.0f;
  int i = tid * GAP;
  for (; i + (UNROLL - 1) * step < n4; i += UNROLL * step) {
    #pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      float4 v = src[i + u * step];
      acc[u] += v.x + v.y + v.z + v.w;
    }
  }
  for (; i < n4; i += step) {
    float4 v = src[i];
    acc[0] += v.x + v.y + v.z + v.w;
  }
  float total = 0.0f;
  for (int u = 0; u < UNROLL; ++u) total += acc[u];
  dst[tid] = total;
}

// ---------------------------------------------------------------------------
// Coalesced store kernels — write iota: dst[i] = i.
// ---------------------------------------------------------------------------

template <int UNROLL>
__global__ void coalesced_store_f32(float *__restrict__ dst, int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  int i = tid;
  for (; i + (UNROLL - 1) * stride < n; i += UNROLL * stride) {
    #pragma unroll
    for (int u = 0; u < UNROLL; ++u)
      dst[i + u * stride] = (float)(i + u * stride);
  }
  for (; i < n; i += stride)
    dst[i] = (float)i;
}

template <int UNROLL>
__global__ void coalesced_store_f32x2(float2 *__restrict__ dst, int n2) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  int i = tid;
  for (; i + (UNROLL - 1) * stride < n2; i += UNROLL * stride) {
    #pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      int idx = i + u * stride;
      int base = idx * 2;
      dst[idx] = make_float2((float)base, (float)(base + 1));
    }
  }
  for (; i < n2; i += stride) {
    int base = i * 2;
    dst[i] = make_float2((float)base, (float)(base + 1));
  }
}

template <int UNROLL>
__global__ void coalesced_store_f32x4(float4 *__restrict__ dst, int n4) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  int i = tid;
  for (; i + (UNROLL - 1) * stride < n4; i += UNROLL * stride) {
    #pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      int idx = i + u * stride;
      int base = idx * 4;
      dst[idx] = make_float4((float)base, (float)(base + 1),
                             (float)(base + 2), (float)(base + 3));
    }
  }
  for (; i < n4; i += stride) {
    int base = i * 4;
    dst[i] = make_float4((float)base, (float)(base + 1),
                         (float)(base + 2), (float)(base + 3));
  }
}

// ---------------------------------------------------------------------------
// Gapped store kernels
// ---------------------------------------------------------------------------

template <int UNROLL, int GAP>
__global__ void gapped_store_f32(float *__restrict__ dst, int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int step = gridDim.x * blockDim.x * GAP;
  int i = tid * GAP;
  for (; i + (UNROLL - 1) * step < n; i += UNROLL * step) {
    #pragma unroll
    for (int u = 0; u < UNROLL; ++u)
      dst[i + u * step] = (float)(i + u * step);
  }
  for (; i < n; i += step)
    dst[i] = (float)i;
}

template <int UNROLL, int GAP>
__global__ void gapped_store_f32x4(float4 *__restrict__ dst, int n4) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int step = gridDim.x * blockDim.x * GAP;
  int i = tid * GAP;
  for (; i + (UNROLL - 1) * step < n4; i += UNROLL * step) {
    #pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      int idx = i + u * step;
      int base = idx * 4;
      dst[idx] = make_float4((float)base, (float)(base + 1),
                             (float)(base + 2), (float)(base + 3));
    }
  }
  for (; i < n4; i += step) {
    int base = i * 4;
    dst[i] = make_float4((float)base, (float)(base + 1),
                         (float)(base + 2), (float)(base + 3));
  }
}

// ---------------------------------------------------------------------------
// Benchmark harness
// ---------------------------------------------------------------------------

static const int WARMUP = 5;
static const int ITERS = 50;

// Returns median GB/s for a single (grid, block) configuration.
template <typename KernelFn>
double bench_one(KernelFn fn, int grid, int block, size_t bytes_moved) {
  std::vector<double> samples(ITERS);

  for (int i = 0; i < WARMUP; ++i)
    fn(grid, block);
  HIP_CHECK(hipDeviceSynchronize());

  hipEvent_t start, stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));

  for (int i = 0; i < ITERS; ++i) {
    flush_cache();
    HIP_CHECK(hipEventRecord(start));
    fn(grid, block);
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    float ms = 0;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    samples[i] = (double)bytes_moved / ((double)ms / 1000.0) / 1e9;
  }

  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));

  std::sort(samples.begin(), samples.end());
  return samples[ITERS / 2];
}

struct BenchResult {
  const char *name;
  double best_gb_per_sec;
  int best_block;
  int best_unroll;
};

// Sweep block sizes and unroll factors, return best config.
// block_sizes: array of block sizes to try.
// For each (block, unroll), we compute num_blocks to fill the GPU.
template <typename LaunchFn>
BenchResult sweep(const char *name, LaunchFn launch_fn,
                  const std::vector<int> &block_sizes, int num_cus,
                  int n, size_t bytes_moved) {
  double best_bw = 0;
  int best_block = 0, best_unroll = 0;

  for (int bs : block_sizes) {
    // Target enough waves to fill the GPU: ~8 waves per CU.
    int waves_per_block = bs / 64;  // conservative (wave64)
    if (waves_per_block < 1) waves_per_block = 1;
    int target_waves = num_cus * 8;
    int num_blocks = (target_waves + waves_per_block - 1) / waves_per_block;
    // Ensure we have enough blocks for the data.
    // Grid-stride handles any N, but more blocks = better occupancy.
    num_blocks = std::max(num_blocks, 256);

    for (int unroll = 1; unroll <= 8; ++unroll) {
      double bw = launch_fn(num_blocks, bs, unroll, n, bytes_moved);
      if (bw > best_bw) {
        best_bw = bw;
        best_block = bs;
        best_unroll = unroll;
      }
    }
  }
  return {name, best_bw, best_block, best_unroll};
}

// ---------------------------------------------------------------------------
// Dispatch helpers — runtime unroll -> compile-time template via switch.
// Use hipLaunchKernelGGL to avoid <<<>>> parsing issues in templates.
// ---------------------------------------------------------------------------

#define DISPATCH_UNROLL(KERNEL, GRID, BLOCK, ...)                              \
  do { switch (unroll) {                                                       \
    case 1: hipLaunchKernelGGL(KERNEL<1>, GRID, BLOCK, 0, 0, __VA_ARGS__); break; \
    case 2: hipLaunchKernelGGL(KERNEL<2>, GRID, BLOCK, 0, 0, __VA_ARGS__); break; \
    case 3: hipLaunchKernelGGL(KERNEL<3>, GRID, BLOCK, 0, 0, __VA_ARGS__); break; \
    case 4: hipLaunchKernelGGL(KERNEL<4>, GRID, BLOCK, 0, 0, __VA_ARGS__); break; \
    case 5: hipLaunchKernelGGL(KERNEL<5>, GRID, BLOCK, 0, 0, __VA_ARGS__); break; \
    case 6: hipLaunchKernelGGL(KERNEL<6>, GRID, BLOCK, 0, 0, __VA_ARGS__); break; \
    case 7: hipLaunchKernelGGL(KERNEL<7>, GRID, BLOCK, 0, 0, __VA_ARGS__); break; \
    case 8: hipLaunchKernelGGL(KERNEL<8>, GRID, BLOCK, 0, 0, __VA_ARGS__); break; \
  } } while(0)

#define DISPATCH_UNROLL_GAP(KERNEL, GAP, GRID, BLOCK, ...)                     \
  do { switch (unroll) {                                                       \
    case 1: hipLaunchKernelGGL((KERNEL<1,GAP>), GRID, BLOCK, 0, 0, __VA_ARGS__); break; \
    case 2: hipLaunchKernelGGL((KERNEL<2,GAP>), GRID, BLOCK, 0, 0, __VA_ARGS__); break; \
    case 3: hipLaunchKernelGGL((KERNEL<3,GAP>), GRID, BLOCK, 0, 0, __VA_ARGS__); break; \
    case 4: hipLaunchKernelGGL((KERNEL<4,GAP>), GRID, BLOCK, 0, 0, __VA_ARGS__); break; \
    case 5: hipLaunchKernelGGL((KERNEL<5,GAP>), GRID, BLOCK, 0, 0, __VA_ARGS__); break; \
    case 6: hipLaunchKernelGGL((KERNEL<6,GAP>), GRID, BLOCK, 0, 0, __VA_ARGS__); break; \
    case 7: hipLaunchKernelGGL((KERNEL<7,GAP>), GRID, BLOCK, 0, 0, __VA_ARGS__); break; \
    case 8: hipLaunchKernelGGL((KERNEL<8,GAP>), GRID, BLOCK, 0, 0, __VA_ARGS__); break; \
  } } while(0)

int main() {
  // Query device properties.
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  int subgroup_size = prop.warpSize;
  int num_cus = prop.multiProcessorCount;

  printf("GPU: %s\n", prop.name);
  printf("CUs: %d, subgroup size: %d\n", num_cus, subgroup_size);

  std::vector<int> block_sizes = {subgroup_size * 4, subgroup_size * 8};
  printf("Block sizes: %d, %d\n", block_sizes[0], block_sizes[1]);
  printf("Unroll factors: 1-8 (best selected automatically)\n\n");

  // Buffer: ~2GB, divisible by largest block * 8 * 4 (float4).
  int max_block = block_sizes.back();
  int align = max_block * 8 * 4;  // block * max_unroll * float4_elems
  const int N = ((512 * 1024 * 1024) / align) * align;  // ~2GB in floats
  const size_t bytes = (size_t)N * sizeof(float);

  printf("Buffer: %.0f MB (%d M floats)\n", (double)bytes / (1024*1024), N / (1024*1024));
  printf("Iterations: %d per config (median), cache flushed between iterations\n\n", ITERS);

  float *d_src, *d_dst;
  HIP_CHECK(hipMalloc(&d_src, bytes));
  HIP_CHECK(hipMalloc(&d_dst, bytes));
  HIP_CHECK(hipMalloc(&d_flush, FLUSH_BYTES));
  HIP_CHECK(hipMemset(d_src, 1, bytes));
  HIP_CHECK(hipMemset(d_dst, 0, bytes));

  std::vector<BenchResult> results;

  // Helper macros: sweep block sizes and unroll factors for each kernel.
  #define SWEEP_COAL_LOAD(NAME, VEC_TYPE, KERNEL, DIVISOR)                     \
    results.push_back(sweep(NAME, [&](int nb, int bs, int unroll,              \
                                      int n, size_t bm) -> double {            \
      return bench_one([&](int g, int b) {                                     \
        DISPATCH_UNROLL(KERNEL, dim3(g), dim3(b), (VEC_TYPE*)d_src, d_dst, n); \
      }, nb, bs, bm);                                                          \
    }, block_sizes, num_cus, N / DIVISOR, bytes))

  #define SWEEP_GAP_LOAD(NAME, KERNEL, GAP, BYTES_MOVED)                      \
    results.push_back(sweep(NAME, [&](int nb, int bs, int unroll,              \
                                      int n, size_t bm) -> double {            \
      return bench_one([&](int g, int b) {                                     \
        DISPATCH_UNROLL_GAP(KERNEL, GAP, dim3(g), dim3(b),                     \
                            (const float*)d_src, d_dst, n);                    \
      }, nb, bs, bm);                                                          \
    }, block_sizes, num_cus, N, BYTES_MOVED))

  #define SWEEP_GAP_LOAD4(NAME, KERNEL, GAP, BYTES_MOVED)                     \
    results.push_back(sweep(NAME, [&](int nb, int bs, int unroll,              \
                                      int n, size_t bm) -> double {            \
      return bench_one([&](int g, int b) {                                     \
        DISPATCH_UNROLL_GAP(KERNEL, GAP, dim3(g), dim3(b),                     \
                            (const float4*)d_src, d_dst, n);                   \
      }, nb, bs, bm);                                                          \
    }, block_sizes, num_cus, N / 4, BYTES_MOVED))

  #define SWEEP_COAL_STORE(NAME, VEC_TYPE, KERNEL, DIVISOR)                    \
    results.push_back(sweep(NAME, [&](int nb, int bs, int unroll,              \
                                      int n, size_t bm) -> double {            \
      return bench_one([&](int g, int b) {                                     \
        DISPATCH_UNROLL(KERNEL, dim3(g), dim3(b), (VEC_TYPE*)d_dst, n);        \
      }, nb, bs, bm);                                                          \
    }, block_sizes, num_cus, N / DIVISOR, bytes))

  #define SWEEP_GAP_STORE(NAME, KERNEL, GAP, BYTES_MOVED)                     \
    results.push_back(sweep(NAME, [&](int nb, int bs, int unroll,              \
                                      int n, size_t bm) -> double {            \
      return bench_one([&](int g, int b) {                                     \
        DISPATCH_UNROLL_GAP(KERNEL, GAP, dim3(g), dim3(b),                     \
                            (float*)d_dst, n);                                 \
      }, nb, bs, bm);                                                          \
    }, block_sizes, num_cus, N, BYTES_MOVED))

  #define SWEEP_GAP_STORE4(NAME, KERNEL, GAP, BYTES_MOVED)                    \
    results.push_back(sweep(NAME, [&](int nb, int bs, int unroll,              \
                                      int n, size_t bm) -> double {            \
      return bench_one([&](int g, int b) {                                     \
        DISPATCH_UNROLL_GAP(KERNEL, GAP, dim3(g), dim3(b),                     \
                            (float4*)d_dst, n);                                \
      }, nb, bs, bm);                                                          \
    }, block_sizes, num_cus, N / 4, BYTES_MOVED))

  printf("Running benchmarks (sweeping block sizes and unroll factors)...\n\n");

  // ==== COALESCED LOADS ====
  SWEEP_COAL_LOAD("float   (b32)  load+reduce", const float,  coalesced_load_f32,   1);
  SWEEP_COAL_LOAD("float2  (b64)  load+reduce", const float2, coalesced_load_f32x2, 2);
  SWEEP_COAL_LOAD("float4  (b128) load+reduce", const float4, coalesced_load_f32x4, 4);

  // ==== GAPPED LOADS ====
  SWEEP_GAP_LOAD("float  stride=2",  gapped_load_f32, 2,  (size_t)(N/2)*4);
  SWEEP_GAP_LOAD("float  stride=4",  gapped_load_f32, 4,  (size_t)(N/4)*4);
  SWEEP_GAP_LOAD("float  stride=32", gapped_load_f32, 32, (size_t)(N/32)*4);
  SWEEP_GAP_LOAD4("float4 stride=2",  gapped_load_f32x4, 2,  (size_t)(N/2)*4);
  SWEEP_GAP_LOAD4("float4 stride=4",  gapped_load_f32x4, 4,  (size_t)(N/4)*4);
  SWEEP_GAP_LOAD4("float4 stride=32", gapped_load_f32x4, 32, (size_t)(N/32)*4);

  // ==== COALESCED STORES ====
  SWEEP_COAL_STORE("float   (b32)  store iota", float,  coalesced_store_f32,   1);
  SWEEP_COAL_STORE("float2  (b64)  store iota", float2, coalesced_store_f32x2, 2);
  SWEEP_COAL_STORE("float4  (b128) store iota", float4, coalesced_store_f32x4, 4);

  // ==== GAPPED STORES ====
  SWEEP_GAP_STORE("float  stride=2",  gapped_store_f32, 2,  (size_t)(N/2)*4);
  SWEEP_GAP_STORE("float  stride=4",  gapped_store_f32, 4,  (size_t)(N/4)*4);
  SWEEP_GAP_STORE("float  stride=32", gapped_store_f32, 32, (size_t)(N/32)*4);
  SWEEP_GAP_STORE4("float4 stride=2",  gapped_store_f32x4, 2,  (size_t)(N/2)*4);
  SWEEP_GAP_STORE4("float4 stride=4",  gapped_store_f32x4, 4,  (size_t)(N/4)*4);
  SWEEP_GAP_STORE4("float4 stride=32", gapped_store_f32x4, 32, (size_t)(N/32)*4);

  // ---- Print results ----
  printf("%-40s %10s %8s %8s\n", "Kernel", "GB/s", "Block", "Unroll");
  printf("%-40s %10s %8s %8s\n",
         "----------------------------------------",
         "----------", "--------", "--------");

  const char *sections[] = {
    "COALESCED LOADS",
    "GAPPED LOADS",
    "COALESCED STORES",
    "GAPPED STORES",
  };
  int counts[] = {3, 6, 3, 6};
  int idx = 0;
  for (int s = 0; s < 4; ++s) {
    printf("\n  == %s ==\n", sections[s]);
    for (int i = 0; i < counts[s]; ++i, ++idx) {
      auto &r = results[idx];
      printf("  %-38s %8.1f %8d %8d\n",
             r.name, r.best_gb_per_sec, r.best_block, r.best_unroll);
    }
  }
  printf("\n");

  HIP_CHECK(hipFree(d_src));
  HIP_CHECK(hipFree(d_dst));
  HIP_CHECK(hipFree(d_flush));
  return 0;
}
