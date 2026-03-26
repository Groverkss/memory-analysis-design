// Coalescing benchmark for RDNA4 (gfx1201)
// Tests load/store bandwidth with varying elements-per-thread and gap patterns.
//
 float *__restrict__ src,
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
// Benchmark harness
// ---------------------------------------------------------------------------

struct BenchResult {
  const char *name;
  double gb_per_sec;
};

template <typename KernelFn>
double bench_kernel(KernelFn fn, int grid, int block, int warmup, int iters,
                    size_t bytes_moved) {
  // Warmup
  for (int i = 0; i < warmup; ++i)
    fn(grid, block);
  HIP_CHECK(hipDeviceSynchronize());

  hipEvent_t start, stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));

  HIP_CHECK(hipEventRecord(start));
  for (int i = 0; i < iters; ++i)
    fn(grid, block);
  HIP_CHECK(hipEventRecord(stop));
  HIP_CHECK(hipEventSynchronize(stop));

  float ms = 0;
  HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));

  double sec = (double)ms / 1000.0 / (double)iters;
  return (double)bytes_moved / sec / 1e9;  // GB/s
}

int main() {
  // 256 MB buffer
  const int N = 64 * 1024 * 1024;  // 64M floats = 256 MB
  const size_t bytes = (size_t)N * sizeof(float);
  const int BLOCK = 256;
  const int WARMUP = 5;
  const int ITERS = 20;

  float *d_src, *d_dst;
  HIP_CHECK(hipMalloc(&d_src, bytes));
  HIP_CHECK(hipMalloc(&d_dst, bytes));
  HIP_CHECK(hipMemset(d_src, 1, bytes));
  HIP_CHECK(hipMemset(d_dst, 0, bytes));

  std::vector<BenchResult> results;

  auto run = [&](const char *name, auto launcher, int grid, size_t moved) {
    double bw = bench_kernel(launcher, grid, BLOCK, WARMUP, ITERS, moved);
    results.push_back({name, bw});
  };

  // ---- Coalesced loads ----
  #define BENCH_COAL_LOAD(EPT)                                                 \
    run("coalesced_load<" #EPT ">",                                            \
        [&](int g, int b) {                                                    \
          coalesced_load<EPT><<<g, b>>>(d_src, d_dst, N);                      \
        },                                                                     \
        N / (EPT * BLOCK), bytes)

  BENCH_COAL_LOAD(1);
  BENCH_COAL_LOAD(2);
  BENCH_COAL_LOAD(4);
  BENCH_COAL_LOAD(8);
  BENCH_COAL_LOAD(16);

  // ---- Gapped loads (GAP=32 means each thread skips a warp-width) ----
  #define BENCH_GAP_LOAD(EPT, GAP)                                             \
    run("gapped_load<" #EPT ",gap=" #GAP ">",                                 \
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
  #define BENCH_COAL_STORE(EPT)                                                \
    run("coalesced_store<" #EPT ">",                                           \
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
  #define BENCH_GAP_STORE(EPT, GAP)                                            \
    run("gapped_store<" #EPT ",gap=" #GAP ">",                                \
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
  printf("\n%-35s %12s\n", "Kernel", "GB/s");
  printf("%-35s %12s\n", "-----------------------------------",
         "------------");
  const char *prev_section = nullptr;
  for (auto &r : results) {
    // Section separator
    const char *section = nullptr;
    if (strstr(r.name, "coalesced_load"))  section = "COALESCED LOADS";
    else if (strstr(r.name, "gapped_load")) section = "GAPPED LOADS";
    else if (strstr(r.name, "coalesced_store")) section = "COALESCED STORES";
    else if (strstr(r.name, "gapped_store")) section = "GAPPED STORES";
    if (section && section != prev_section) {
      printf("\n  == %s ==\n", section);
      prev_section = section;
    }
    printf("%-35s %10.1f\n", r.name, r.gb_per_sec);
  }
  printf("\n");

  HIP_CHECK(hipFree(d_src));
  HIP_CHECK(hipFree(d_dst));
  return 0;
}
