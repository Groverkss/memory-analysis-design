// Coalescing benchmark — f16 element type
//
// Tests load/store bandwidth with varying vector widths and gap patterns.
// For each kernel type, sweeps block sizes (subgroup*4, subgroup*8) and
// unroll factors (1-8), reports the best configuration.
//
// Load kernels: grid-stride loop, batch reduce into float accumulators.
// Store kernels: grid-stride loop, write iota.
//
// Vector widths (128-bit max = coalescing limit):
//   half    = 1x f16 =  2B  -> global_load_b16
//   half2   = 2x f16 =  4B  -> global_load_b32
//   half4   = 4x f16 =  8B  -> global_load_b64
//   half8   = 8x f16 = 16B  -> global_load_b128

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <algorithm>
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

// Aligned vector types for f16 that guarantee the right load widths.
typedef _Float16 half1_t;

// 2x f16 = 4 bytes, aligned to 4B -> global_load_b32
typedef struct __attribute__((aligned(4))) { _Float16 x, y; } half2_t;

// 4x f16 = 8 bytes, aligned to 8B -> global_load_b64
typedef struct __attribute__((aligned(8))) { _Float16 x, y, z, w; } half4_t;

// 8x f16 = 16 bytes, aligned to 16B -> global_load_b128
typedef struct __attribute__((aligned(16))) {
  _Float16 a, b, c, d, e, f, g, h;
} half8_t;

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
// Coalesced load kernels — accumulate into float to measure memory not ALU.
// ---------------------------------------------------------------------------

// 1x f16 = 2B per load (global_load_b16)
template <int UNROLL>
__global__ void coalesced_load_f16x1(const half1_t *__restrict__ src,
                                     float *__restrict__ dst, int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  float acc[UNROLL];
  for (int u = 0; u < UNROLL; ++u) acc[u] = 0.0f;
  int i = tid;
  for (; i + (UNROLL - 1) * stride < n; i += UNROLL * stride) {
    #pragma unroll
    for (int u = 0; u < UNROLL; ++u)
      acc[u] += (float)src[i + u * stride];
  }
  for (; i < n; i += stride)
    acc[0] += (float)src[i];
  float total = 0.0f;
  for (int u = 0; u < UNROLL; ++u) total += acc[u];
  dst[tid] = total;
}

// 2x f16 = 4B per load (global_load_b32)
template <int UNROLL>
__global__ void coalesced_load_f16x2(const half2_t *__restrict__ src,
                                     float *__restrict__ dst, int n2) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  float acc[UNROLL];
  for (int u = 0; u < UNROLL; ++u) acc[u] = 0.0f;
  int i = tid;
  for (; i + (UNROLL - 1) * stride < n2; i += UNROLL * stride) {
    #pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      half2_t v = src[i + u * stride];
      acc[u] += (float)v.x + (float)v.y;
    }
  }
  for (; i < n2; i += stride) {
    half2_t v = src[i];
    acc[0] += (float)v.x + (float)v.y;
  }
  float total = 0.0f;
  for (int u = 0; u < UNROLL; ++u) total += acc[u];
  dst[tid] = total;
}

// 4x f16 = 8B per load (global_load_b64)
template <int UNROLL>
__global__ void coalesced_load_f16x4(const half4_t *__restrict__ src,
                                     float *__restrict__ dst, int n4) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  float acc[UNROLL];
  for (int u = 0; u < UNROLL; ++u) acc[u] = 0.0f;
  int i = tid;
  for (; i + (UNROLL - 1) * stride < n4; i += UNROLL * stride) {
    #pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      half4_t v = src[i + u * stride];
      acc[u] += (float)v.x + (float)v.y + (float)v.z + (float)v.w;
    }
  }
  for (; i < n4; i += stride) {
    half4_t v = src[i];
    acc[0] += (float)v.x + (float)v.y + (float)v.z + (float)v.w;
  }
  float total = 0.0f;
  for (int u = 0; u < UNROLL; ++u) total += acc[u];
  dst[tid] = total;
}

// 8x f16 = 16B per load (global_load_b128)
template <int UNROLL>
__global__ void coalesced_load_f16x8(const half8_t *__restrict__ src,
                                     float *__restrict__ dst, int n8) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  float acc[UNROLL];
  for (int u = 0; u < UNROLL; ++u) acc[u] = 0.0f;
  int i = tid;
  for (; i + (UNROLL - 1) * stride < n8; i += UNROLL * stride) {
    #pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      half8_t v = src[i + u * stride];
      acc[u] += (float)v.a + (float)v.b + (float)v.c + (float)v.d +
                (float)v.e + (float)v.f + (float)v.g + (float)v.h;
    }
  }
  for (; i < n8; i += stride) {
    half8_t v = src[i];
    acc[0] += (float)v.a + (float)v.b + (float)v.c + (float)v.d +
              (float)v.e + (float)v.f + (float)v.g + (float)v.h;
  }
  float total = 0.0f;
  for (int u = 0; u < UNROLL; ++u) total += acc[u];
  dst[tid] = total;
}

// ---------------------------------------------------------------------------
// Gapped load kernels
// ---------------------------------------------------------------------------

template <int UNROLL, int GAP>
__global__ void gapped_load_f16x1(const half1_t *__restrict__ src,
                                  float *__restrict__ dst, int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int step = gridDim.x * blockDim.x * GAP;
  float acc[UNROLL];
  for (int u = 0; u < UNROLL; ++u) acc[u] = 0.0f;
  int i = tid * GAP;
  for (; i + (UNROLL - 1) * step < n; i += UNROLL * step) {
    #pragma unroll
    for (int u = 0; u < UNROLL; ++u)
      acc[u] += (float)src[i + u * step];
  }
  for (; i < n; i += step)
    acc[0] += (float)src[i];
  float total = 0.0f;
  for (int u = 0; u < UNROLL; ++u) total += acc[u];
  dst[tid] = total;
}

template <int UNROLL, int GAP>
__global__ void gapped_load_f16x8(const half8_t *__restrict__ src,
                                  float *__restrict__ dst, int n8) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int step = gridDim.x * blockDim.x * GAP;
  float acc[UNROLL];
  for (int u = 0; u < UNROLL; ++u) acc[u] = 0.0f;
  int i = tid * GAP;
  for (; i + (UNROLL - 1) * step < n8; i += UNROLL * step) {
    #pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      half8_t v = src[i + u * step];
      acc[u] += (float)v.a + (float)v.b + (float)v.c + (float)v.d +
                (float)v.e + (float)v.f + (float)v.g + (float)v.h;
    }
  }
  for (; i < n8; i += step) {
    half8_t v = src[i];
    acc[0] += (float)v.a + (float)v.b + (float)v.c + (float)v.d +
              (float)v.e + (float)v.f + (float)v.g + (float)v.h;
  }
  float total = 0.0f;
  for (int u = 0; u < UNROLL; ++u) total += acc[u];
  dst[tid] = total;
}

// ---------------------------------------------------------------------------
// Coalesced store kernels — write iota as f16.
// ---------------------------------------------------------------------------

template <int UNROLL>
__global__ void coalesced_store_f16x1(half1_t *__restrict__ dst, int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  int i = tid;
  for (; i + (UNROLL - 1) * stride < n; i += UNROLL * stride) {
    #pragma unroll
    for (int u = 0; u < UNROLL; ++u)
      dst[i + u * stride] = (_Float16)(i + u * stride);
  }
  for (; i < n; i += stride)
    dst[i] = (_Float16)i;
}

template <int UNROLL>
__global__ void coalesced_store_f16x2(half2_t *__restrict__ dst, int n2) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  int i = tid;
  for (; i + (UNROLL - 1) * stride < n2; i += UNROLL * stride) {
    #pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      int idx = i + u * stride;
      int base = idx * 2;
      dst[idx] = {(_Float16)base, (_Float16)(base + 1)};
    }
  }
  for (; i < n2; i += stride) {
    int base = i * 2;
    dst[i] = {(_Float16)base, (_Float16)(base + 1)};
  }
}

template <int UNROLL>
__global__ void coalesced_store_f16x4(half4_t *__restrict__ dst, int n4) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  int i = tid;
  for (; i + (UNROLL - 1) * stride < n4; i += UNROLL * stride) {
    #pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      int idx = i + u * stride;
      int base = idx * 4;
      dst[idx] = {(_Float16)base, (_Float16)(base+1),
                  (_Float16)(base+2), (_Float16)(base+3)};
    }
  }
  for (; i < n4; i += stride) {
    int base = i * 4;
    dst[i] = {(_Float16)base, (_Float16)(base+1),
              (_Float16)(base+2), (_Float16)(base+3)};
  }
}

template <int UNROLL>
__global__ void coalesced_store_f16x8(half8_t *__restrict__ dst, int n8) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  int i = tid;
  for (; i + (UNROLL - 1) * stride < n8; i += UNROLL * stride) {
    #pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      int idx = i + u * stride;
      int base = idx * 8;
      dst[idx] = {(_Float16)base, (_Float16)(base+1), (_Float16)(base+2),
                  (_Float16)(base+3), (_Float16)(base+4), (_Float16)(base+5),
                  (_Float16)(base+6), (_Float16)(base+7)};
    }
  }
  for (; i < n8; i += stride) {
    int base = i * 8;
    dst[i] = {(_Float16)base, (_Float16)(base+1), (_Float16)(base+2),
              (_Float16)(base+3), (_Float16)(base+4), (_Float16)(base+5),
              (_Float16)(base+6), (_Float16)(base+7)};
  }
}

// ---------------------------------------------------------------------------
// Gapped store kernels
// ---------------------------------------------------------------------------

template <int UNROLL, int GAP>
__global__ void gapped_store_f16x1(half1_t *__restrict__ dst, int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int step = gridDim.x * blockDim.x * GAP;
  int i = tid * GAP;
  for (; i + (UNROLL - 1) * step < n; i += UNROLL * step) {
    #pragma unroll
    for (int u = 0; u < UNROLL; ++u)
      dst[i + u * step] = (_Float16)(i + u * step);
  }
  for (; i < n; i += step)
    dst[i] = (_Float16)i;
}

template <int UNROLL, int GAP>
__global__ void gapped_store_f16x8(half8_t *__restrict__ dst, int n8) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int step = gridDim.x * blockDim.x * GAP;
  int i = tid * GAP;
  for (; i + (UNROLL - 1) * step < n8; i += UNROLL * step) {
    #pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      int idx = i + u * step;
      int base = idx * 8;
      dst[idx] = {(_Float16)base, (_Float16)(base+1), (_Float16)(base+2),
                  (_Float16)(base+3), (_Float16)(base+4), (_Float16)(base+5),
                  (_Float16)(base+6), (_Float16)(base+7)};
    }
  }
  for (; i < n8; i += step) {
    int base = i * 8;
    dst[i] = {(_Float16)base, (_Float16)(base+1), (_Float16)(base+2),
              (_Float16)(base+3), (_Float16)(base+4), (_Float16)(base+5),
              (_Float16)(base+6), (_Float16)(base+7)};
  }
}

// ---------------------------------------------------------------------------
// Benchmark harness
// ---------------------------------------------------------------------------

static const int WARMUP = 5;
static const int ITERS = 50;

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

template <typename LaunchFn>
BenchResult sweep(const char *name, LaunchFn launch_fn,
                  const std::vector<int> &block_sizes, int num_cus,
                  int n, size_t bytes_moved) {
  double best_bw = 0;
  int best_block = 0, best_unroll = 0;

  for (int bs : block_sizes) {
    int waves_per_block = bs / 64;
    if (waves_per_block < 1) waves_per_block = 1;
    int target_waves = num_cus * 8;
    int num_blocks = (target_waves + waves_per_block - 1) / waves_per_block;
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
// Dispatch macros
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
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  int subgroup_size = prop.warpSize;
  int num_cus = prop.multiProcessorCount;

  printf("GPU: %s\n", prop.name);
  printf("CUs: %d, subgroup size: %d\n", num_cus, subgroup_size);

  std::vector<int> block_sizes = {subgroup_size * 4, subgroup_size * 8};
  printf("Block sizes: %d, %d\n", block_sizes[0], block_sizes[1]);
  printf("Unroll factors: 1-8 (best selected automatically)\n\n");

  // Buffer in f16 elements. ~2GB total.
  int max_block = block_sizes.back();
  int align = max_block * 8 * 8;  // block * max_unroll * max_vector(8)
  const int N = ((1024 * 1024 * 1024) / align) * align;  // ~2GB in f16 elems
  const size_t bytes = (size_t)N * sizeof(_Float16);

  printf("Element type: f16 (2 bytes)\n");
  printf("Buffer: %.0f MB (%d M f16 elements)\n",
         (double)bytes / (1024*1024), N / (1024*1024));
  printf("Iterations: %d per config (median), cache flushed between iterations\n\n", ITERS);

  _Float16 *d_src;
  float *d_dst;  // accumulators are float
  HIP_CHECK(hipMalloc(&d_src, bytes));
  // dst only needs one float per thread; allocate generously
  HIP_CHECK(hipMalloc(&d_dst, (size_t)max_block * 8 * 1024 * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_flush, FLUSH_BYTES));
  HIP_CHECK(hipMemset(d_src, 1, bytes));
  HIP_CHECK(hipMemset(d_dst, 0, (size_t)max_block * 8 * 1024 * sizeof(float)));

  std::vector<BenchResult> results;

  // Sweep macros
  #define SWEEP_COAL_LOAD(NAME, PTR_TYPE, KERNEL, DIVISOR)                     \
    results.push_back(sweep(NAME, [&](int nb, int bs, int unroll,              \
                                      int n, size_t bm) -> double {            \
      return bench_one([&](int g, int b) {                                     \
        DISPATCH_UNROLL(KERNEL, dim3(g), dim3(b),                              \
                        (const PTR_TYPE*)d_src, d_dst, n);                     \
      }, nb, bs, bm);                                                          \
    }, block_sizes, num_cus, N / DIVISOR, bytes))

  #define SWEEP_COAL_STORE(NAME, PTR_TYPE, KERNEL, DIVISOR)                    \
    results.push_back(sweep(NAME, [&](int nb, int bs, int unroll,              \
                                      int n, size_t bm) -> double {            \
      return bench_one([&](int g, int b) {                                     \
        DISPATCH_UNROLL(KERNEL, dim3(g), dim3(b),                              \
                        (PTR_TYPE*)d_src, n);                                  \
      }, nb, bs, bm);                                                          \
    }, block_sizes, num_cus, N / DIVISOR, bytes))

  #define SWEEP_GAP_LOAD(NAME, KERNEL, GAP, BYTES)                            \
    results.push_back(sweep(NAME, [&](int nb, int bs, int unroll,              \
                                      int n, size_t bm) -> double {            \
      return bench_one([&](int g, int b) {                                     \
        DISPATCH_UNROLL_GAP(KERNEL, GAP, dim3(g), dim3(b),                     \
                            (const half1_t*)d_src, d_dst, n);                  \
      }, nb, bs, bm);                                                          \
    }, block_sizes, num_cus, N, BYTES))

  #define SWEEP_GAP_LOAD8(NAME, KERNEL, GAP, BYTES)                           \
    results.push_back(sweep(NAME, [&](int nb, int bs, int unroll,              \
                                      int n, size_t bm) -> double {            \
      return bench_one([&](int g, int b) {                                     \
        DISPATCH_UNROLL_GAP(KERNEL, GAP, dim3(g), dim3(b),                     \
                            (const half8_t*)d_src, d_dst, n);                  \
      }, nb, bs, bm);                                                          \
    }, block_sizes, num_cus, N / 8, BYTES))

  #define SWEEP_GAP_STORE(NAME, KERNEL, GAP, BYTES)                           \
    results.push_back(sweep(NAME, [&](int nb, int bs, int unroll,              \
                                      int n, size_t bm) -> double {            \
      return bench_one([&](int g, int b) {                                     \
        DISPATCH_UNROLL_GAP(KERNEL, GAP, dim3(g), dim3(b),                     \
                            (half1_t*)d_src, n);                               \
      }, nb, bs, bm);                                                          \
    }, block_sizes, num_cus, N, BYTES))

  #define SWEEP_GAP_STORE8(NAME, KERNEL, GAP, BYTES)                          \
    results.push_back(sweep(NAME, [&](int nb, int bs, int unroll,              \
                                      int n, size_t bm) -> double {            \
      return bench_one([&](int g, int b) {                                     \
        DISPATCH_UNROLL_GAP(KERNEL, GAP, dim3(g), dim3(b),                     \
                            (half8_t*)d_src, n);                               \
      }, nb, bs, bm);                                                          \
    }, block_sizes, num_cus, N / 8, BYTES))

  printf("Running benchmarks (sweeping block sizes and unroll factors)...\n\n");

  // ==== COALESCED LOADS ====
  SWEEP_COAL_LOAD("half    (1x f16 =  2B) -> b16",  half1_t, coalesced_load_f16x1, 1);
  SWEEP_COAL_LOAD("half2   (2x f16 =  4B) -> b32",  half2_t, coalesced_load_f16x2, 2);
  SWEEP_COAL_LOAD("half4   (4x f16 =  8B) -> b64",  half4_t, coalesced_load_f16x4, 4);
  SWEEP_COAL_LOAD("half8   (8x f16 = 16B) -> b128", half8_t, coalesced_load_f16x8, 8);

  // ==== GAPPED LOADS ====
  SWEEP_GAP_LOAD("half   stride=2",  gapped_load_f16x1, 2,  (size_t)(N/2)*2);
  SWEEP_GAP_LOAD("half   stride=4",  gapped_load_f16x1, 4,  (size_t)(N/4)*2);
  SWEEP_GAP_LOAD("half   stride=32", gapped_load_f16x1, 32, (size_t)(N/32)*2);
  SWEEP_GAP_LOAD8("half8  stride=2",  gapped_load_f16x8, 2,  (size_t)(N/2)*2);
  SWEEP_GAP_LOAD8("half8  stride=4",  gapped_load_f16x8, 4,  (size_t)(N/4)*2);
  SWEEP_GAP_LOAD8("half8  stride=32", gapped_load_f16x8, 32, (size_t)(N/32)*2);

  // ==== COALESCED STORES ====
  SWEEP_COAL_STORE("half    (1x f16 =  2B) -> b16",  half1_t, coalesced_store_f16x1, 1);
  SWEEP_COAL_STORE("half2   (2x f16 =  4B) -> b32",  half2_t, coalesced_store_f16x2, 2);
  SWEEP_COAL_STORE("half4   (4x f16 =  8B) -> b64",  half4_t, coalesced_store_f16x4, 4);
  SWEEP_COAL_STORE("half8   (8x f16 = 16B) -> b128", half8_t, coalesced_store_f16x8, 8);

  // ==== GAPPED STORES ====
  SWEEP_GAP_STORE("half   stride=2",  gapped_store_f16x1, 2,  (size_t)(N/2)*2);
  SWEEP_GAP_STORE("half   stride=4",  gapped_store_f16x1, 4,  (size_t)(N/4)*2);
  SWEEP_GAP_STORE("half   stride=32", gapped_store_f16x1, 32, (size_t)(N/32)*2);
  SWEEP_GAP_STORE8("half8  stride=2",  gapped_store_f16x8, 2,  (size_t)(N/2)*2);
  SWEEP_GAP_STORE8("half8  stride=4",  gapped_store_f16x8, 4,  (size_t)(N/4)*2);
  SWEEP_GAP_STORE8("half8  stride=32", gapped_store_f16x8, 32, (size_t)(N/32)*2);

  // ---- Print results ----
  printf("%-45s %10s %8s %8s\n", "Kernel", "GB/s", "Block", "Unroll");
  printf("%-45s %10s %8s %8s\n",
         "---------------------------------------------",
         "----------", "--------", "--------");

  const char *sections[] = {
    "COALESCED LOADS",
    "GAPPED LOADS",
    "COALESCED STORES",
    "GAPPED STORES",
  };
  int counts[] = {4, 6, 4, 6};
  int idx = 0;
  for (int s = 0; s < 4; ++s) {
    printf("\n  == %s ==\n", sections[s]);
    for (int i = 0; i < counts[s]; ++i, ++idx) {
      auto &r = results[idx];
      printf("  %-43s %8.1f %8d %8d\n",
             r.name, r.best_gb_per_sec, r.best_block, r.best_unroll);
    }
  }
  printf("\n");

  HIP_CHECK(hipFree(d_src));
  HIP_CHECK(hipFree(d_dst));
  HIP_CHECK(hipFree(d_flush));
  return 0;
}
