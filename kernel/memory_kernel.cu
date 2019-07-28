#include <stdio.h>

__global__ void init_ptr_chase(size_t *p, int buf_size, int stride) {
  int stride_num = buf_size / stride;
  if (stride_num <= 1) {
    return;
  }
  for (int i = 0; i < stride_num - 1; ++i) {
    p[i * stride / sizeof(size_t)] = (size_t)(p + (i + 1) * stride / sizeof(size_t));
    //printf("%p, %p\n", &(p[i * stride / sizeof(size_t)]), p[i * stride / sizeof(size_t)]);
    //std::cerr << &(p[i * stride / sizeof(size_t)]) << " " <<  p[i * stride / sizeof(size_t)] << std::endl;
  }
  p[(stride_num - 1) * stride / sizeof(size_t)] = (size_t)p;
  //printf("%p, %p\n", &(p[(stride_num - 1) * stride / sizeof(size_t)]), p[(stride_num - 1) * stride / sizeof(size_t)]);
}

__global__ void ldr_to_use_latency(int *out, size_t *in) {
  size_t access_num = 1024 * 1024 * 64;
  size_t *p = in;
  for (size_t i = 0; i < access_num; ++i) {
    size_t tmp = *p;
    p = (size_t*)tmp;
  } 
  *out = (int)*p;
} 
