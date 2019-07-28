#include "common.h"
#include "bench.h"

int measure_fma_lldr_coissue(bool verbose, int mhz) {
  const size_t block_num = 8192;
  const size_t threads_per_block = 1024;
  const size_t fma_per_thread = 64 * 64;
  const long long total_fma = block_num * threads_per_block * fma_per_thread;
  float *p;
  cudaMalloc(&p, sizeof(float) * block_num * threads_per_block);
  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, NULL);
  float scale = 1.0f;
  fma_lldr_coissue<<<block_num, threads_per_block>>>(p, scale);
  int err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "failed to execute kernel" << std::endl;
  }

  cudaEventRecord(stop, NULL);
  cudaEventSynchronize(stop);

  float ellapse_ms = 0.0f;
  cudaEventElapsedTime(&ellapse_ms, start, stop);
  cudaFree(p);
  if (verbose) {
    std::cerr << ellapse_ms << " ms" << std::endl;
    std::cerr << static_cast<double>(ellapse_ms) * 1e-3 * mhz * 1e6 * 5 * 128 / total_fma << std::endl;
    std::cerr << 2.0 * (total_fma * 1.0e-9) / (ellapse_ms * 1e-3) << " gflops" << std::endl;
  }
  return 0;
}

int measure_fmax2_lldr_coissue(bool verbose, int mhz) {
  const size_t block_num = 8192;
  const size_t threads_per_block = 1024;
  const size_t fma_per_thread = 64 * 64 * 2;
  const long long total_fma = block_num * threads_per_block * fma_per_thread;
  float *p;
  cudaMalloc(&p, sizeof(float) * block_num * threads_per_block);
  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, NULL);
  float scale = 1.0f;
  fmax2_lldr_coissue<<<block_num, threads_per_block>>>(p, scale, scale);
  int err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "failed to execute kernel" << std::endl;
  }

  cudaEventRecord(stop, NULL);
  cudaEventSynchronize(stop);

  float ellapse_ms = 0.0f;
  cudaEventElapsedTime(&ellapse_ms, start, stop);
  cudaFree(p);
  if (verbose) {
    std::cerr << ellapse_ms << " ms" << std::endl;
    std::cerr << static_cast<double>(ellapse_ms) * 1e-3 * mhz * 1e6 * 5 * 128 / total_fma << std::endl;
    std::cerr << 2.0 * (total_fma * 1.0e-9) / (ellapse_ms * 1e-3) << " gflops" << std::endl;
  }
  return 0;
}

int measure_fmax4_lldr_coissue(bool verbose, int mhz) {
  const size_t block_num = 8192;
  const size_t threads_per_block = 1024;
  const size_t fma_per_thread = 64 * 64 * 4;
  const long long total_fma = block_num * threads_per_block * fma_per_thread;
  float *p;
  cudaMalloc(&p, sizeof(float) * block_num * threads_per_block);
  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, NULL);
  float scale = 1.0f;
  fmax4_lldr_coissue<<<block_num, threads_per_block>>>(p, scale, scale, scale, scale);
  int err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "failed to execute kernel" << std::endl;
  }

  cudaEventRecord(stop, NULL);
  cudaEventSynchronize(stop);

  float ellapse_ms = 0.0f;
  cudaEventElapsedTime(&ellapse_ms, start, stop);
  cudaFree(p);
  if (verbose) {
    std::cerr << ellapse_ms << " ms" << std::endl;
    std::cerr << static_cast<double>(ellapse_ms) * 1e-3 * mhz * 1e6 * 5 * 128 / total_fma << std::endl;
    std::cerr << 2.0 * (total_fma * 1.0e-9) / (ellapse_ms * 1e-3) << " gflops" << std::endl;
  }
  return 0;
}

#ifdef LOCAL_MAIN
int main(void) {
  // measure_fmax2_lldr_coissue(true, 1176);
  measure_fmax4_lldr_coissue(true, 1176);
  return 0;
}

#endif
