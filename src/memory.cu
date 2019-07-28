#include "common.h"
#include "bench.h"


/*
void init_ptr_chase(size_t *p, int buf_size, int stride) {
  int stride_num = buf_size / stride;
  if (stride_num <= 1) {
    return;
  }
  for (int i = 0; i < stride_num - 1; ++i) {
    p[i * stride / sizeof(size_t)] = (size_t)(p + (i + 1) * stride / sizeof(size_t));
  }
  p[(stride_num - 1) * stride / sizeof(size_t)] = (size_t)p;
}
*/

int measure_memory_latency(bool verbose, int freq) {
  const int block_num = 1;
  const int threads_per_block = 1;
  size_t access_num = 1024 * 1024 * 64;
  int buf_kb_size[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192}; 
  int stride[] = {16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144};
  for (int i = 0; i < sizeof(buf_kb_size) / sizeof(int); ++i) {
    for (int j = 0; j < sizeof(stride) / sizeof(int); ++j) {
      int buf_size = buf_kb_size[i] * 1024;
      int cur_stride = stride[j];
      if (buf_size >= 2 * cur_stride) {
        size_t *p;
        cudaMalloc(&p, buf_size); // allocate 32M memory
        int *out;
        cudaMalloc(&out, 128); // allocate 128B memory
        init_ptr_chase<<<1, 1>>>(p, buf_size, cur_stride);
        int err = cudaGetLastError();
        if (err != cudaSuccess) {
          std::cerr << "failed to init pointer chase " << err << std::endl;
          return -1;
        }
        cudaEvent_t start;
        cudaEvent_t stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, NULL);
        ldr_to_use_latency<<<block_num, threads_per_block>>>(out, p);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
          std::cerr << "failed to execute kernel" << err << std::endl;
          return -1;
        }
        cudaEventRecord(stop, NULL);
        cudaEventSynchronize(stop);
        float ellapse_ms = 0.0f;
        cudaEventElapsedTime(&ellapse_ms, start, stop);
        if (verbose) {
          std::cerr << 1.0 * (1e3 * freq * ellapse_ms) / static_cast<double>(access_num) << ",";
        }
        cudaFree(p);
        cudaFree(out);
      } else {
        std::cerr << "N/A,";
      }
    }
    std::cerr << std::endl;
  }
  return 0;
}

#ifdef LOCAL_MAIN
int main(void) {

  measure_memory_latency(true, 1176);

}

#endif
