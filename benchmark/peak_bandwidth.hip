// Peak memory bandwidth measurement via a simple copy kernel.
// Compile: hipcc --offload-arch=<arch> -O3 peak_bandwidth.hip -o peak_bandwidth
// Run:     ./peak_bandwidth

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

#define HIP_CHECK(cmd)                                                         \
  do {                                                                         \
    hipError_t err = (cmd);                                                    \
    if (err != hipSuccess) {                                                   \
      fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(err),      \
              __FILE__, __LINE__);                                             \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// 256 MB — large enough to avoid cache effects.
constexpr size_t kBytes = 256ULL * 1024 * 1024;
constexpr int kWarmup = 5;
constexpr int kIterations = 20;

__global__ void copy_kernel(const float4 *__restrict__ src,
                            float4 *__restrict__ dst, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    dst[i] = src[i];
}

int main() {
  hipDeviceProp_t props;
  HIP_CHECK(hipGetDeviceProperties(&props, 0));
  printf("Device: %s\n", props.name);
  printf("Copy size: %zu MB\n", kBytes / (1024 * 1024));

  float4 *d_src, *d_dst;
  HIP_CHECK(hipMalloc(&d_src, kBytes));
  HIP_CHECK(hipMalloc(&d_dst, kBytes));
  HIP_CHECK(hipMemset(d_src, 1, kBytes));

  size_t n = kBytes / sizeof(float4);
  int threads = 256;
  int blocks = (n + threads - 1) / threads;

  // Warmup.
  for (int i = 0; i < kWarmup; i++)
    copy_kernel<<<blocks, threads>>>(d_src, d_dst, n);
  HIP_CHECK(hipDeviceSynchronize());

  // Timed iterations.
  hipEvent_t start, stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));

  HIP_CHECK(hipEventRecord(start));
  for (int i = 0; i < kIterations; i++)
    copy_kernel<<<blocks, threads>>>(d_src, d_dst, n);
  HIP_CHECK(hipEventRecord(stop));
  HIP_CHECK(hipEventSynchronize(stop));

  float ms = 0;
  HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
  double seconds = ms / 1000.0 / kIterations;
  // Read + write = 2x the buffer size.
  double bw_gbps = (2.0 * kBytes) / seconds / 1e9;

  printf("Avg time: %.3f us\n", seconds * 1e6);
  printf("PEAK_BW_GBPS=%.1f\n", bw_gbps);

  HIP_CHECK(hipFree(d_src));
  HIP_CHECK(hipFree(d_dst));
  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));
  return 0;
}
