
/*
__global__ void shared_memory_load_throughput(float *c_buffer) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ float shared_buffer[256];
  if (threadIdx.x < 256) {
    shared_buffer[threadIdx.x] = (float)threadIdx.x;
  }
  __syncthreads();


  float c = (float)index;
  for (int j = 0; j < 32; ++j) {
    for (int i = 0; i < 256; ++i) {
      c += shared_buffer[i];
    }
  }
  c_buffer[index] = c;
}
*/

#define BUF_SIZE 4096

__global__ void shared_memory_latency(int *out, const int stride) {
  __shared__ size_t shared_buf[BUF_SIZE / sizeof(size_t)];
  int stride_num = BUF_SIZE / stride;
  if (stride_num <= 1) {
    return;
  }
  for (int i = 0; i < stride_num - 1; ++i) {
    shared_buf[i * stride / sizeof(size_t)] = (size_t)(shared_buf + (i + 1) * stride / sizeof(size_t));
  }
  shared_buf[(stride_num - 1) * stride / sizeof(size_t)] = (size_t)shared_buf;
  size_t *p = shared_buf;
  size_t access_num = 1024 * 1024 * 64;
  for (size_t i = 0; i < access_num; ++i) {
    p = (size_t*)*p;
  } 
  *out = (int)*p;
} 
