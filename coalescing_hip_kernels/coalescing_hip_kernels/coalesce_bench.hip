// Coalescing benchmark — f16, row-major vs column-major 3D layout
//
// Input: P x M x N f16 elements, reduce over M and N per workgroup.
// P = num_cus (one workgroup per CU).
// N = subgroup_size * vec_elems (one coalesced vector load per thread per row).
// M = number of rows (large).
//
// Each subgroup independently reduces M rows of N elements.
// Block = subgroup_size * 4 (4 subgroups per workgroup, each independent).
//
// Row-major:  mat[P][M][N] — workgroup p's slice is contiguous.
//             Stride between rows = N (small). Cache friendly.
//
// Col-major:  mat[M][P][N] — workgroup p's rows interleaved with other WGs.
//             Stride between rows = P*N (large). Cache hostile.
//
// Both are coalesced: adjacent threads in a subgroup load adjacent elements.

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

typedef _Float16 half1_t;
typedef struct __attribute__((aligned(4)))  { _Float16 x, y; } half2_t;
typedef struct __attribute__((aligned(8)))  { _Float16 x, y, z, w; } half4_t;
typedef struct __attribute__((aligned(16))) {
  _Float16 a, b, c, d, e, f, g, h;
} half8_t;

__device__ __forceinline__ float hsum(half1_t v) { return (float)v; }
__device__ __forceinline__ float hsum(half2_t v) {
  return (float)v.x + (float)v.y;
}
__device__ __forceinline__ float hsum(half4_t v) {
  return (float)v.x + (float)v.y + (float)v.z + (float)v.w;
}
__device__ __forceinline__ float hsum(half8_t v) {
  return (float)v.a + (float)v.b + (float)v.c + (float)v.d +
         (float)v.e + (float)v.f + (float)v.g + (float)v.h;
}

// ---------------------------------------------------------------------------
// Cache flush
// ---------------------------------------------------------------------------
static float *d_flush = nullptr;
static const size_t FLUSH_BYTES = 512ULL * 1024 * 1024;

__global__ void flush_kernel(float *buf, int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) buf[tid] = (float)tid;
}

static void flush_cache() {
  int n = FLUSH_BYTES / sizeof(float);
  flush_kernel<<<(n + 255) / 256, 256>>>(d_flush, n);
  HIP_CHECK(hipDeviceSynchronize());
}

// ===================================================================
// ROW-MAJOR LOAD: mat[P][M][N] — workgroup p reads mat[p] as flat.
//
// Workgroup p's slice is MN_vec contiguous vec elements. Grid-stride
// within the block over the flat slice. Sequential memory access.
// ===================================================================

#define MAKE_ROWMAJOR_LOAD(SUFFIX, VEC_TYPE)                                   \
  template <int UNROLL>                                                        \
  __global__ void rowmajor_load_##SUFFIX(const VEC_TYPE *__restrict__ src,     \
                                         float *__restrict__ dst,              \
                                         int MN_vec) {                         \
    const VEC_TYPE *my_slice = src + blockIdx.x * MN_vec;                      \
    int tid = threadIdx.x;                                                     \
    int stride = blockDim.x;                                                   \
    float acc[UNROLL];                                                         \
    for (int u = 0; u < UNROLL; ++u) acc[u] = 0.0f;                           \
    int i = tid;                                                               \
    for (; i + (UNROLL-1) * stride < MN_vec; i += UNROLL * stride) {           \
      _Pragma("unroll")                                                        \
      for (int u = 0; u < UNROLL; ++u)                                         \
        acc[u] += hsum(my_slice[i + u * stride]);                              \
    }                                                                          \
    for (; i < MN_vec; i += stride)                                            \
      acc[0] += hsum(my_slice[i]);                                             \
    float t = 0.0f;                                                            \
    for (int u = 0; u < UNROLL; ++u) t += acc[u];                              \
    dst[blockIdx.x * blockDim.x + threadIdx.x] = t;                           \
  }

MAKE_ROWMAJOR_LOAD(f16x1, half1_t)
MAKE_ROWMAJOR_LOAD(f16x2, half2_t)
MAKE_ROWMAJOR_LOAD(f16x4, half4_t)
MAKE_ROWMAJOR_LOAD(f16x8, half8_t)

// ===================================================================
// COL-MAJOR LOAD: mat[M][P][N] — workgroup p reads mat[0..M-1][p][*].
//
// Each subgroup loads one row per step. Subgroups stride over M rows.
//   lane_id = threadIdx.x % subgroup_size
//   subgroup_id = threadIdx.x / subgroup_size
//   num_subgroups = blockDim.x / subgroup_size
//
// Row stride = P * N_vec (large). Unroll along M (the strided dim).
// ===================================================================

#define MAKE_COLMAJOR_LOAD(SUFFIX, VEC_TYPE)                                   \
  template <int UNROLL>                                                        \
  __global__ void colmajor_load_##SUFFIX(const VEC_TYPE *__restrict__ src,     \
                                         float *__restrict__ dst,              \
                                         int M, int N_vec, int P) {            \
    int lane = threadIdx.x % N_vec;                                            \
    int sg_id = threadIdx.x / N_vec;                                           \
    int num_sg = blockDim.x / N_vec;                                           \
    int row_stride = P * N_vec;                                                \
    const VEC_TYPE *my_start = src + blockIdx.x * N_vec;                       \
    float acc[UNROLL];                                                         \
    for (int u = 0; u < UNROLL; ++u) acc[u] = 0.0f;                           \
    int row = sg_id;                                                           \
    for (; row + (UNROLL-1) * num_sg < M; row += UNROLL * num_sg) {            \
      _Pragma("unroll")                                                        \
      for (int u = 0; u < UNROLL; ++u)                                         \
        acc[u] += hsum(my_start[(row + u * num_sg) * row_stride + lane]);      \
    }                                                                          \
    for (; row < M; row += num_sg)                                             \
      acc[0] += hsum(my_start[row * row_stride + lane]);                       \
    float t = 0.0f;                                                            \
    for (int u = 0; u < UNROLL; ++u) t += acc[u];                              \
    dst[blockIdx.x * blockDim.x + threadIdx.x] = t;                           \
  }

MAKE_COLMAJOR_LOAD(f16x1, half1_t)
MAKE_COLMAJOR_LOAD(f16x2, half2_t)
MAKE_COLMAJOR_LOAD(f16x4, half4_t)
MAKE_COLMAJOR_LOAD(f16x8, half8_t)

// ===================================================================
// LANE REMAPPING VARIANTS — test whether lane order within a subgroup
// affects coalescing. All variants touch the same set of addresses per
// subgroup, just with different thread-to-address mappings.
//
// Normal:   lane = threadIdx.x % N_vec  (identity)
// Reversed: lane = (N_vec - 1) - (threadIdx.x % N_vec)
// XOR:      lane = (threadIdx.x % N_vec) ^ (N_vec / 2)  (swap halves)
// ===================================================================

// Row-major with lane remapping: same flat grid-stride but remap
// threadIdx within each subgroup.
#define MAKE_ROWMAJOR_LOAD_REMAP(SUFFIX, VEC_TYPE, LANE_EXPR)                  \
  template <int UNROLL>                                                        \
  __global__ void rowmajor_load_##SUFFIX(const VEC_TYPE *__restrict__ src,     \
                                         float *__restrict__ dst,              \
                                         int MN_vec, int N_vec) {              \
    const VEC_TYPE *my_slice = src + blockIdx.x * MN_vec;                      \
    int raw_lane = threadIdx.x % N_vec;                                        \
    int sg_base = threadIdx.x - raw_lane;                                      \
    int lane = LANE_EXPR;                                                      \
    int tid = sg_base + lane;                                                  \
    int stride = blockDim.x;                                                   \
    float acc[UNROLL];                                                         \
    for (int u = 0; u < UNROLL; ++u) acc[u] = 0.0f;                           \
    int i = tid;                                                               \
    for (; i + (UNROLL-1) * stride < MN_vec; i += UNROLL * stride) {           \
      _Pragma("unroll")                                                        \
      for (int u = 0; u < UNROLL; ++u)                                         \
        acc[u] += hsum(my_slice[i + u * stride]);                              \
    }                                                                          \
    for (; i < MN_vec; i += stride)                                            \
      acc[0] += hsum(my_slice[i]);                                             \
    float t = 0.0f;                                                            \
    for (int u = 0; u < UNROLL; ++u) t += acc[u];                              \
    dst[blockIdx.x * blockDim.x + threadIdx.x] = t;                           \
  }

MAKE_ROWMAJOR_LOAD_REMAP(f16x8_reversed, half8_t,
                          (N_vec - 1) - raw_lane)
MAKE_ROWMAJOR_LOAD_REMAP(f16x8_xor_half, half8_t,
                          raw_lane ^ (N_vec / 2))

// Col-major with lane remapping.
#define MAKE_COLMAJOR_LOAD_REMAP(SUFFIX, VEC_TYPE, LANE_EXPR)                  \
  template <int UNROLL>                                                        \
  __global__ void colmajor_load_##SUFFIX(const VEC_TYPE *__restrict__ src,     \
                                         float *__restrict__ dst,              \
                                         int M, int N_vec, int P) {            \
    int raw_lane = threadIdx.x % N_vec;                                        \
    int lane = LANE_EXPR;                                                      \
    int sg_id = threadIdx.x / N_vec;                                           \
    int num_sg = blockDim.x / N_vec;                                           \
    int row_stride = P * N_vec;                                                \
    const VEC_TYPE *my_start = src + blockIdx.x * N_vec;                       \
    float acc[UNROLL];                                                         \
    for (int u = 0; u < UNROLL; ++u) acc[u] = 0.0f;                           \
    int row = sg_id;                                                           \
    for (; row + (UNROLL-1) * num_sg < M; row += UNROLL * num_sg) {            \
      _Pragma("unroll")                                                        \
      for (int u = 0; u < UNROLL; ++u)                                         \
        acc[u] += hsum(my_start[(row + u * num_sg) * row_stride + lane]);      \
    }                                                                          \
    for (; row < M; row += num_sg)                                             \
      acc[0] += hsum(my_start[row * row_stride + lane]);                       \
    float t = 0.0f;                                                            \
    for (int u = 0; u < UNROLL; ++u) t += acc[u];                              \
    dst[blockIdx.x * blockDim.x + threadIdx.x] = t;                           \
  }

MAKE_COLMAJOR_LOAD_REMAP(f16x8_reversed, half8_t,
                          (N_vec - 1) - raw_lane)
MAKE_COLMAJOR_LOAD_REMAP(f16x8_xor_half, half8_t,
                          raw_lane ^ (N_vec / 2))

// ===================================================================
// ROW-MAJOR STORE: mat[P][M][N] — workgroup p writes mat[p] as flat.
// ===================================================================

#define MAKE_ROWMAJOR_STORE(SUFFIX, VEC_TYPE, MAKE_VAL)                        \
  template <int UNROLL>                                                        \
  __global__ void rowmajor_store_##SUFFIX(VEC_TYPE *__restrict__ dst,          \
                                          int MN_vec) {                        \
    VEC_TYPE *my_slice = dst + blockIdx.x * MN_vec;                            \
    int tid = threadIdx.x;                                                     \
    int stride = blockDim.x;                                                   \
    int i = tid;                                                               \
    for (; i + (UNROLL-1) * stride < MN_vec; i += UNROLL * stride) {           \
      _Pragma("unroll")                                                        \
      for (int u = 0; u < UNROLL; ++u) {                                       \
        int idx = i + u * stride;                                              \
        int v = idx ^ tid;                                                     \
        my_slice[idx] = MAKE_VAL;                                              \
      }                                                                        \
    }                                                                          \
    for (; i < MN_vec; i += stride) {                                          \
      int v = i ^ tid;                                                         \
      my_slice[i] = MAKE_VAL;                                                  \
    }                                                                          \
  }

// v = idx ^ threadIdx.x — cheap, unique per thread and per iteration.
#define MAKE_H1(v) ((_Float16)(v))
#define MAKE_H2(v) {(_Float16)(v), (_Float16)((v) ^ 1)}
#define MAKE_H4(v) {(_Float16)(v), (_Float16)((v) ^ 1), \
                    (_Float16)((v) ^ 2), (_Float16)((v) ^ 3)}
#define MAKE_H8(v) {(_Float16)(v), (_Float16)((v) ^ 1), (_Float16)((v) ^ 2), \
                    (_Float16)((v) ^ 3), (_Float16)((v) ^ 4), (_Float16)((v) ^ 5), \
                    (_Float16)((v) ^ 6), (_Float16)((v) ^ 7)}

MAKE_ROWMAJOR_STORE(f16x1, half1_t, MAKE_H1(v))
MAKE_ROWMAJOR_STORE(f16x2, half2_t, MAKE_H2(v))
MAKE_ROWMAJOR_STORE(f16x4, half4_t, MAKE_H4(v))
MAKE_ROWMAJOR_STORE(f16x8, half8_t, MAKE_H8(v))

// ===================================================================
// COL-MAJOR STORE: mat[M][P][N] — workgroup p writes mat[0..M-1][p][*].
// ===================================================================

#define MAKE_COLMAJOR_STORE(SUFFIX, VEC_TYPE, MAKE_VAL)                        \
  template <int UNROLL>                                                        \
  __global__ void colmajor_store_##SUFFIX(VEC_TYPE *__restrict__ dst,          \
                                          int M, int N_vec, int P) {           \
    int lane = threadIdx.x % N_vec;                                            \
    int sg_id = threadIdx.x / N_vec;                                           \
    int num_sg = blockDim.x / N_vec;                                           \
    int row_stride = P * N_vec;                                                \
    VEC_TYPE *my_start = dst + blockIdx.x * N_vec;                             \
    int tid = threadIdx.x;                                                     \
    int row = sg_id;                                                           \
    for (; row + (UNROLL-1) * num_sg < M; row += UNROLL * num_sg) {            \
      _Pragma("unroll")                                                        \
      for (int u = 0; u < UNROLL; ++u) {                                       \
        int idx = (row + u * num_sg) * row_stride + lane;                      \
        int v = idx ^ tid;                                                     \
        my_start[idx] = MAKE_VAL;                                              \
      }                                                                        \
    }                                                                          \
    for (; row < M; row += num_sg) {                                           \
      int idx = row * row_stride + lane;                                       \
      int v = idx ^ tid;                                                       \
      my_start[idx] = MAKE_VAL;                                                \
    }                                                                          \
  }

MAKE_COLMAJOR_STORE(f16x1, half1_t, MAKE_H1(v))
MAKE_COLMAJOR_STORE(f16x2, half2_t, MAKE_H2(v))
MAKE_COLMAJOR_STORE(f16x4, half4_t, MAKE_H4(v))
MAKE_COLMAJOR_STORE(f16x8, half8_t, MAKE_H8(v))

// ---------------------------------------------------------------------------
// Benchmark harness
// ---------------------------------------------------------------------------

static const int WARMUP = 3;
static const int ITERS = 10;

template <typename KernelFn>
double bench_one(KernelFn fn, int grid, int block, size_t bytes_moved) {
  std::vector<double> samples(ITERS);
  for (int i = 0; i < WARMUP; ++i) fn(grid, block);
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
                  const std::vector<int> &block_sizes, int P,
                  size_t bytes_moved) {
  double best_bw = 0;
  int best_block = 0, best_unroll = 0;
  for (int bs : block_sizes) {
    for (int unroll = 1; unroll <= 8; ++unroll) {
      double bw = launch_fn(P, bs, unroll, bytes_moved);
      if (bw > best_bw) {
        best_bw = bw;
        best_block = bs;
        best_unroll = unroll;
      }
    }
  }
  return {name, best_bw, best_block, best_unroll};
}

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

int main() {
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  int subgroup_size = prop.warpSize;
  int num_cus = prop.multiProcessorCount;
  int P = 4 * num_cus;

  printf("GPU: %s\n", prop.name);
  printf("CUs: %d, subgroup size: %d\n", num_cus, subgroup_size);

  std::vector<int> block_sizes = {subgroup_size * 4, subgroup_size * 8};
  int N_vec = subgroup_size;  // vec elements per row (1 vec load per thread)

  printf("Block sizes: %d, %d\n", block_sizes[0], block_sizes[1]);
  printf("Unroll factors: 1-8 (best selected automatically)\n");
  printf("P (workgroups) = %d (4 * num_cus)\n\n", P);

  // M*N (in f16 elements) is constant across vector widths.
  // N_f16 = subgroup_size * vec_elems. M = MN / N_f16.
  // Size buffer so total ~2GB: P * MN * 2 bytes = 2GB => MN = 2GB / (P * 2).
  int MN = (int)((2ULL * 1024 * 1024 * 1024) / ((size_t)P * sizeof(_Float16)));
  // Round down to multiple of subgroup_size * 8 * 8 for all vec widths + unroll.
  int align = subgroup_size * 8 * 8;
  MN = (MN / align) * align;
  size_t bytes = (size_t)P * MN * sizeof(_Float16);

  // Per vector width: N_f16 = subgroup_size * vec_elems, M = MN / N_f16.
  int M1 = MN / (subgroup_size * 1);
  int M2 = MN / (subgroup_size * 2);
  int M4 = MN / (subgroup_size * 4);
  int M8 = MN / (subgroup_size * 8);

  printf("Element type: f16 (2 bytes)\n");
  printf("M*N per workgroup = %d f16 elements (constant)\n", MN);
  printf("Per vector width (N = subgroup_size * vec_elems, M = MN/N):\n");
  printf("  half1: N=%3d, M=%d\n", subgroup_size * 1, M1);
  printf("  half2: N=%3d, M=%d\n", subgroup_size * 2, M2);
  printf("  half4: N=%3d, M=%d\n", subgroup_size * 4, M4);
  printf("  half8: N=%3d, M=%d\n", subgroup_size * 8, M8);
  printf("Buffer: %.0f MB (P=%d workgroups)\n", (double)bytes / (1024*1024), P);
  printf("Row-major: mat[P][M][N] — WG slice contiguous\n");
  printf("Col-major: mat[M][P][N] — WG stride = P*N between rows\n");
  printf("Iterations: %d per config (median), cache flushed between iterations\n\n",
         ITERS);

  _Float16 *d_buf;
  float *d_dst;
  HIP_CHECK(hipMalloc(&d_buf, bytes));
  HIP_CHECK(hipMalloc(&d_dst,
             (size_t)P * block_sizes.back() * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_flush, FLUSH_BYTES));
  HIP_CHECK(hipMemset(d_buf, 1, bytes));

  std::vector<BenchResult> results;

  // Sweep macros. Each vec width has its own M but same MN f16 elements.
  // MN_vec = MN / vec_elems = total vec elements per workgroup.
  // bytes_moved = P * MN * sizeof(_Float16) = bytes (same for all).
  #define SWEEP_RM_LOAD(NAME, KERNEL, PTR_TYPE, VEC_ELEMS, M_VAL)              \
    results.push_back(sweep(NAME,                                              \
      [&](int p, int bs, int unroll, size_t bm) -> double {                    \
        int mn_vec = MN / VEC_ELEMS;                                           \
        return bench_one([&](int g, int b) {                                   \
          DISPATCH_UNROLL(KERNEL, dim3(g), dim3(b),                            \
                          (const PTR_TYPE*)d_buf, d_dst, mn_vec);              \
        }, p, bs, bytes);                                                      \
      }, block_sizes, P, 0))

  #define SWEEP_CM_LOAD(NAME, KERNEL, PTR_TYPE, VEC_ELEMS, M_VAL)              \
    results.push_back(sweep(NAME,                                              \
      [&](int p, int bs, int unroll, size_t bm) -> double {                    \
        return bench_one([&](int g, int b) {                                   \
          DISPATCH_UNROLL(KERNEL, dim3(g), dim3(b),                            \
                          (const PTR_TYPE*)d_buf, d_dst, M_VAL, N_vec, p);     \
        }, p, bs, bytes);                                                      \
      }, block_sizes, P, 0))

  #define SWEEP_RM_STORE(NAME, KERNEL, PTR_TYPE, VEC_ELEMS, M_VAL)             \
    results.push_back(sweep(NAME,                                              \
      [&](int p, int bs, int unroll, size_t bm) -> double {                    \
        int mn_vec = MN / VEC_ELEMS;                                           \
        return bench_one([&](int g, int b) {                                   \
          DISPATCH_UNROLL(KERNEL, dim3(g), dim3(b),                            \
                          (PTR_TYPE*)d_buf, mn_vec);                           \
        }, p, bs, bytes);                                                      \
      }, block_sizes, P, 0))

  #define SWEEP_CM_STORE(NAME, KERNEL, PTR_TYPE, VEC_ELEMS, M_VAL)             \
    results.push_back(sweep(NAME,                                              \
      [&](int p, int bs, int unroll, size_t bm) -> double {                    \
        return bench_one([&](int g, int b) {                                   \
          DISPATCH_UNROLL(KERNEL, dim3(g), dim3(b),                            \
                          (PTR_TYPE*)d_buf, M_VAL, N_vec, p);                  \
        }, p, bs, bytes);                                                      \
      }, block_sizes, P, 0))

  printf("Running benchmarks...\n\n");

  SWEEP_RM_LOAD("half1 (b16)  row-major", rowmajor_load_f16x1, half1_t, 1, M1);
  SWEEP_RM_LOAD("half2 (b32)  row-major", rowmajor_load_f16x2, half2_t, 2, M2);
  SWEEP_RM_LOAD("half4 (b64)  row-major", rowmajor_load_f16x4, half4_t, 4, M4);
  SWEEP_RM_LOAD("half8 (b128) row-major", rowmajor_load_f16x8, half8_t, 8, M8);

  SWEEP_CM_LOAD("half1 (b16)  col-major", colmajor_load_f16x1, half1_t, 1, M1);
  SWEEP_CM_LOAD("half2 (b32)  col-major", colmajor_load_f16x2, half2_t, 2, M2);
  SWEEP_CM_LOAD("half4 (b64)  col-major", colmajor_load_f16x4, half4_t, 4, M4);
  SWEEP_CM_LOAD("half8 (b128) col-major", colmajor_load_f16x8, half8_t, 8, M8);

  // ==== LANE REMAPPING (half8 only) ====
  // Row-major remapped: takes (src, dst, MN_vec, N_vec)
  #define SWEEP_RM_REMAP(NAME, KERNEL)                                         \
    results.push_back(sweep(NAME,                                              \
      [&](int p, int bs, int unroll, size_t bm) -> double {                    \
        int mn_vec = MN / 8;                                                   \
        return bench_one([&](int g, int b) {                                   \
          DISPATCH_UNROLL(KERNEL, dim3(g), dim3(b),                            \
                          (const half8_t*)d_buf, d_dst, mn_vec, N_vec);        \
        }, p, bs, bytes);                                                      \
      }, block_sizes, P, 0))

  // Col-major remapped: same signature as normal col-major
  #define SWEEP_CM_REMAP(NAME, KERNEL)                                         \
    results.push_back(sweep(NAME,                                              \
      [&](int p, int bs, int unroll, size_t bm) -> double {                    \
        return bench_one([&](int g, int b) {                                   \
          DISPATCH_UNROLL(KERNEL, dim3(g), dim3(b),                            \
                          (const half8_t*)d_buf, d_dst, M8, N_vec, p);         \
        }, p, bs, bytes);                                                      \
      }, block_sizes, P, 0))

  SWEEP_RM_REMAP("half8 row-major reversed",  rowmajor_load_f16x8_reversed);
  SWEEP_RM_REMAP("half8 row-major xor-half",  rowmajor_load_f16x8_xor_half);
  SWEEP_CM_REMAP("half8 col-major reversed",  colmajor_load_f16x8_reversed);
  SWEEP_CM_REMAP("half8 col-major xor-half",  colmajor_load_f16x8_xor_half);

  SWEEP_RM_STORE("half1 (b16)  row-major", rowmajor_store_f16x1, half1_t, 1, M1);
  SWEEP_RM_STORE("half2 (b32)  row-major", rowmajor_store_f16x2, half2_t, 2, M2);
  SWEEP_RM_STORE("half4 (b64)  row-major", rowmajor_store_f16x4, half4_t, 4, M4);
  SWEEP_RM_STORE("half8 (b128) row-major", rowmajor_store_f16x8, half8_t, 8, M8);

  SWEEP_CM_STORE("half1 (b16)  col-major", colmajor_store_f16x1, half1_t, 1, M1);
  SWEEP_CM_STORE("half2 (b32)  col-major", colmajor_store_f16x2, half2_t, 2, M2);
  SWEEP_CM_STORE("half4 (b64)  col-major", colmajor_store_f16x4, half4_t, 4, M4);
  SWEEP_CM_STORE("half8 (b128) col-major", colmajor_store_f16x8, half8_t, 8, M8);

  // ---- Print results ----
  printf("%-40s %10s %8s %8s\n", "Kernel", "GB/s", "Block", "Unroll");
  printf("%-40s %10s %8s %8s\n",
         "----------------------------------------",
         "----------", "--------", "--------");

  const char *sections[] = {
    "ROW-MAJOR LOADS  (mat[P][M][N], stride=N between rows)",
    "COL-MAJOR LOADS  (mat[M][P][N], stride=P*N between rows)",
    "LANE REMAPPING LOADS (half8 only, same addresses, different mapping)",
    "ROW-MAJOR STORES (mat[P][M][N], stride=N between rows)",
    "COL-MAJOR STORES (mat[M][P][N], stride=P*N between rows)",
  };
  int counts[] = {4, 4, 4, 4, 4};
  int idx = 0;
  for (int s = 0; s < 5; ++s) {
    printf("\n  == %s ==\n", sections[s]);
    for (int i = 0; i < counts[s]; ++i, ++idx) {
      auto &r = results[idx];
      printf("  %-38s %8.1f %8d %8d\n",
             r.name, r.best_gb_per_sec, r.best_block, r.best_unroll);
    }
  }
  printf("\n");

  HIP_CHECK(hipFree(d_buf));
  HIP_CHECK(hipFree(d_dst));
  HIP_CHECK(hipFree(d_flush));
  return 0;
}
