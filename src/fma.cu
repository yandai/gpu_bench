#include "common.h"
#include "bench.h"

#define MHZ 1176
#define CUDA_CORE_NUM 640

int measure_fma_throughput(bool verbose, int mhz) {
  const size_t block_num = 8192;
  const size_t threads_per_block = 1024;
  const size_t fma_per_thread = 2048 * 16;
  const long long total_fma = block_num * threads_per_block * fma_per_thread;
  float *out;
  cudaMalloc(&out, sizeof(float) * block_num * threads_per_block);
  float scale[16];
  for (int i = 0; i < 16; ++i) {
    scale[i] = i;
  }
  float *d_scale;
  cudaMalloc(&d_scale, sizeof(float) * 16);
  cudaMemcpy(d_scale, scale, sizeof(float) * 16,
        cudaMemcpyHostToDevice); 
  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, NULL);
  fma_throughput<<<block_num, threads_per_block>>>(out, d_scale);
  int err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "failed to execute kernel" << std::endl;
  }

  cudaEventRecord(stop, NULL);
  cudaEventSynchronize(stop);

  float ellapse_ms = 0.0f;
  cudaEventElapsedTime(&ellapse_ms, start, stop);
  cudaFree(out);
  cudaFree(d_scale);
  if (verbose) {
    int cuda_core_num = CUDA_CORE_NUM;
    std::cerr << ellapse_ms << " ms" << std::endl;
    std::cerr << 2.0 * (total_fma * 1.0e-9) / (ellapse_ms * 1e-3) << " gflops" << std::endl;
    std::cerr << "cycle per inst " << ellapse_ms * 1e-3 * mhz * 1e6 * cuda_core_num / (total_fma) << std::endl;
  }
  return 0;
}

int measure_fma_latency(bool verbose, int mhz) {
  const size_t block_num = 1;
  const size_t threads_per_block = 32;
  const size_t fma_per_thread = 1024 * 1024 * 1024;
  const long long total_fma = block_num * threads_per_block * fma_per_thread;
  float *p;
  cudaMalloc(&p, sizeof(float) * block_num * threads_per_block);
  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, NULL);
  float scale = 1.0f;
  fma_latency<<<block_num, threads_per_block>>>(p, scale);
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
    std::cerr << "fma latency cpi" << ellapse_ms * 1e-3 * mhz * 1e6 / fma_per_thread << std::endl;
    
  }
  return 0;
}

#ifdef LOCAL_MAIN
int main(void) {

  measure_fma_latency(true, MHZ);
  measure_fma_throughput(true, MHZ);

}

#endif
