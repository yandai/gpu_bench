#include "common.h"
#include "bench.h"


/*
int measure_shared_memory_peak(int freq, bool verbose) {
  const int block_num = 1024;
  const int threads_per_block = 256;
  const int shared_load_per_thread = 8192;
  float *p;
  cudaMalloc(&p, sizeof(float) * block_num * threads_per_block);
  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, NULL);

  shared_memory_load_throughput<<<block_num, threads_per_block>>>(p);
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
    std::cerr << 1.0 * sizeof(float) * block_num * threads_per_block * shared_load_per_thread / (1e3 * freq * ellapse_ms) / 5.0
              << " byte/cycle per sm" << std::endl;
  }
  return 0;
}
*/


int measure_shared_memory_latency(bool verbose, int freq) {
  const int block_num = 1;
  const int threads_per_block = 1;
  size_t access_num = 1024 * 1024 * 64;
  int *p;
  cudaMalloc(&p, sizeof(int) * block_num * threads_per_block);
  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, NULL);
  shared_memory_latency<<<block_num, threads_per_block>>>(p, 64);
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
    std::cerr << "ellapse_time " << ellapse_ms << "ms" << std::endl;
    std::cerr << 1.0 * (1e3 * freq * ellapse_ms) / static_cast<double>(access_num) 
              << " cycle latency per load" << std::endl;
  }
  return 0;
}

#ifdef LOCAL_MAIN
int main(void) {

  measure_shared_memory_latency(true, 1176);

}

#endif
