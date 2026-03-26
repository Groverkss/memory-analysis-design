// Coalescing benchmark
// Tests load/store bandwidth with varying elements-per-thread and gap patterns.
//
// Each kernel processes a buffer of N floats. We launch enough threads so that
// every element is touched exactly once (coalesced) or with a known stride
// (gapped). We measure effective bandwidth = bytes_moved / time.

#include <hip/hip_runtime.h>
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

// ---------------------------------------------------------------------------
// Cache flush: write garbage to a separate buffer that's large enough to
// evict all L1/L2 contents. MI300X has 256MB L2, RDNA4 has 64MB L2.
// We use 512MB to be safe on all targets.
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
// Coalesced load: each thread loads EPT contiguous elements, accumulates them,
// and writes one result to prevent dead-code elimination.
// Thread i reads elements [i*EPT, i*EPT + EPT).
// ---------------------------------------------------------------------------
template <int EPT>
__global__ void coalesced_load(const float *__restrict__ src,
                               float *__restrict__ dst, int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int base = tid * EPT;
  if (base + EPT > n) return;
  float acc = 0.0f;
  #pragma unroll
  for (int i = 0; i < EPT; ++i)
    acc += src[base + i];
  dst[tid] = acc;
}

// ---------------------------------------------------------------------------
// Gapped load: each thread loads EPT elements with a stride of GAP between
// them. Thread i reads elements [i*EPT*GAP, ...] stepping by GAP.
// This creates intra-warp gaps — consecutive threads still start at adjacent
// EPT*GAP blocks, but within each thread the accesses are strided.
// ---------------------------------------------------------------------------
template <int EPT, int GAP>
__global__ void gapped_load(const float *__restrict__ src,
                            float *__restrict__ dst, int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int base = tid * EPT * GAP;
  if (base + (EPT - 1) * GAP >= n) return;
  float acc = 0.0f;
  #pragma unroll
  for (int i = 0; i < EPT; ++i)
    acc += src[base + i * GAP];
  dst[tid] = acc;
}

// ---------------------------------------------------------------------------
// Coalesced store: each thread writes EPT contiguous elements.
// Thread i writes to [i*EPT, i*EPT + EPT).
// ---------------------------------------------------------------------------
template <int EPT>
__global__ void coalesced_store(float *__restrict__ dst, int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int base = tid * EPT;
  if (base + EPT > n) return;
  float val = (float)tid;
  #pragma unroll
  for (int i = 0; i < EPT; ++i)
    dst[base + i] = val + (float)i;
}

// ---------------------------------------------------------------------------
// Gapped store: each thread writes EPT elements with stride GAP.
// ---------------------------------------------------------------------------
template <int EPT, int GAP>
__global__ void gapped_store(float *__restrict__ dst, int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int base = tid * EPT * GAP;
  if (base + (EPT - 1) * GAP >= n) return;
  float val = (float)tid;
  #pragma unroll
  for (int i = 0; i < EPT; ++i)
    dst[base + i * GAP] = val + (float)i;
}

// ---------------------------------------------------------------------------
// Benchmark harness — runs ITERS individual timed launches, takes median.
// Flushes cache before each iteration.
// ---------------------------------------------------------------------------

struct BenchResult {
  const char *name;
  double median_gb_per_sec;
  double min_gb_per_sec;
  double max_gb_per_sec;
};

template <typename KernelFn>
BenchResult bench_kernel(const char *name, KernelFn fn, int grid, int block,
                         int warmup, int iters, size_t bytes_moved) {
  std::vector<double> samples(iters);

  // Warmup (no cache flush — just get the GPU clocks up)
  for (int i = 0; i < warmup; ++i) {
    fn(grid, block);
  }
  HIP_CHECK(hipDeviceSynchronize());

  hipEvent_t start, stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));

  for (int i = 0; i < iters; ++i) {
    // Flush cache before each iteration
    flush_cache();

    HIP_CHECK(hipEventRecord(start));
    fn(grid, block);
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));

    float ms = 0;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    double sec = (double)ms / 1000.0;
    samples[i] = (double)bytes_moved / sec / 1e9;
  }

  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));

  std::sort(samples.begin(), samples.end());
  double median = samples[iters / 2];
  double min_bw = samples[0];
  double max_bw = samples[iters - 1];

  return {name, median, min_bw, max_bw};
}

int main() {
  // 256 MB buffer
  const int N = 64 * 1024 * 1024;  // 64M floats = 256 MB
  const size_t bytes = (size_t)N * sizeof(float);
  const int BLOCK = 256;
  const int WARMUP = 10;
  const int ITERS = 50;

  float *d_src, *d_dst;
  HIP_CHECK(hipMalloc(&d_src, bytes));
  HIP_CHECK(hipMalloc(&d_dst, bytes));
  HIP_CHECK(hipMalloc(&d_flush, FLUSH_BYTES));
  HIP_CHECK(hipMemset(d_src, 1, bytes));
  HIP_CHECK(hipMemset(d_dst, 0, bytes));

  std::vector<BenchResult> results;

  auto run = [&](const char *name, auto launcher, int grid, size_t moved) {
    auto r = bench_kernel(name, launcher, grid, BLOCK, WARMUP, ITERS, moved);
    results.push_back(r);
  };

  // EPT = elements per thread. Each element is a f32 (4 bytes).
  // EPT=1  -> each thread does 1x f32 load/store  =  4B  (global_load_b32)
  // EPT=2  -> each thread does 2x f32 load/store  =  8B  (global_load_b64)
  // EPT=4  -> each thread does 4x f32 load/store  = 16B  (global_load_b128)
  // EPT=8  -> each thread does 8x f32 load/store  = 32B  (2x global_load_b128)
  // EPT=16 -> each thread does 16x f32 load/store = 64B  (4x global_load_b128)
  //
  // GAP = stride between elements within a thread. GAP=1 is contiguous.
  // GAP=2 means thread reads every other element (50% cache line utilization).
  // GAP=32 means thread reads one element per cache line (worst case).

  // ---- Coalesced loads ----
  // Each thread loads EPT contiguous f32s. Adjacent threads access adjacent
  // blocks, so the entire warp's access pattern is fully coalesced.
  #define BENCH_COAL_LOAD(EPT)                                                 \
    run(#EPT "x f32 = " #EPT "*4B contiguous",                                \
        [&](int g, int b) {                                                    \
          coalesced_load<EPT><<<g, b>>>(d_src, d_dst, N);                      \
        },                                                                     \
        N / (EPT * BLOCK), bytes)

  BENCH_COAL_LOAD(1);
  BENCH_COAL_LOAD(2);
  BENCH_COAL_LOAD(4);
  BENCH_COAL_LOAD(8);
  BENCH_COAL_LOAD(16);

  // ---- Gapped loads ----
  // Each thread loads EPT f32s but with GAP elements between each access.
  // E.g., EPT=2, GAP=4: thread reads src[base] and src[base+4], skipping 3.
  #define BENCH_GAP_LOAD(EPT, GAP)                                             \
    run(#EPT "x f32 stride=" #GAP " (" #EPT "*4B/thread)",                    \
        [&](int g, int b) {                                                    \
          gapped_load<EPT, GAP><<<g, b>>>(d_src, d_dst, N);                   \
        },                                                                     \
        N / (EPT * GAP * BLOCK),                                               \
        (size_t)(N / (EPT * GAP * BLOCK)) * BLOCK * EPT * sizeof(float))

  BENCH_GAP_LOAD(1, 2);
  BENCH_GAP_LOAD(1, 4);
  BENCH_GAP_LOAD(1, 32);
  BENCH_GAP_LOAD(2, 2);
  BENCH_GAP_LOAD(2, 4);
  BENCH_GAP_LOAD(2, 32);
  BENCH_GAP_LOAD(4, 2);
  BENCH_GAP_LOAD(4, 4);
  BENCH_GAP_LOAD(4, 32);

  // ---- Coalesced stores ----
  // Each thread writes EPT contiguous f32s.
  #define BENCH_COAL_STORE(EPT)                                                \
    run(#EPT "x f32 = " #EPT "*4B contiguous",                                \
        [&](int g, int b) {                                                    \
          coalesced_store<EPT><<<g, b>>>(d_dst, N);                            \
        },                                                                     \
        N / (EPT * BLOCK), bytes)

  BENCH_COAL_STORE(1);
  BENCH_COAL_STORE(2);
  BENCH_COAL_STORE(4);
  BENCH_COAL_STORE(8);
  BENCH_COAL_STORE(16);

  // ---- Gapped stores ----
  // Each thread writes EPT f32s with GAP elements between each access.
  #define BENCH_GAP_STORE(EPT, GAP)                                            \
    run(#EPT "x f32 stride=" #GAP " (" #EPT "*4B/thread)",                    \
        [&](int g, int b) {                                                    \
          gapped_store<EPT, GAP><<<g, b>>>(d_dst, N);                         \
        },                                                                     \
        N / (EPT * GAP * BLOCK),                                               \
        (size_t)(N / (EPT * GAP * BLOCK)) * BLOCK * EPT * sizeof(float))

  BENCH_GAP_STORE(1, 2);
  BENCH_GAP_STORE(1, 4);
  BENCH_GAP_STORE(1, 32);
  BENCH_GAP_STORE(2, 2);
  BENCH_GAP_STORE(2, 4);
  BENCH_GAP_STORE(2, 32);
  BENCH_GAP_STORE(4, 2);
  BENCH_GAP_STORE(4, 4);
  BENCH_GAP_STORE(4, 32);

  // ---- Print results ----
  printf("\nElement type: f32 (4 bytes)\n");
  printf("Buffer: %d MB (%dM floats)\n", (int)(bytes / (1024*1024)), N / (1024*1024));
  printf("Block size: %d threads, Iterations: %d (median reported)\n\n", BLOCK, ITERS);

  printf("%-40s %10s %10s %10s\n", "Kernel", "Median", "Min", "Max");
  printf("%-40s %10s %10s %10s\n",
         "----------------------------------------",
         "----------", "----------", "----------");
  const char *prev_section = nullptr;
  for (auto &r : results) {
    const char *section = nullptr;
    if (strstr(r.name, "contiguous") && !prev_section)
      section = "COALESCED LOADS (contiguous f32s per thread)";
    else if (strstr(r.name, "stride") && (!prev_section || strstr(prev_section, "COALESCED LOADS")))
      section = "GAPPED LOADS (strided f32s per thread)";
    else if (strstr(r.name, "contiguous") && strstr(prev_section, "GAPPED LOADS"))
      section = "COALESCED STORES (contiguous f32s per thread)";
    else if (strstr(r.name, "stride") && strstr(prev_section, "COALESCED STORES"))
      section = "GAPPED STORES (strided f32s per thread)";
    if (section) {
      printf("\n  == %s ==\n", section);
      prev_section = section;
    }
    printf("  %-38s %8.1f %8.1f %8.1f\n",
           r.name, r.median_gb_per_sec, r.min_gb_per_sec, r.max_gb_per_sec);
  }
  printf("\n");

  HIP_CHECK(hipFree(d_src));
  HIP_CHECK(hipFree(d_dst));
  HIP_CHECK(hipFree(d_flush));
  return 0;
}
