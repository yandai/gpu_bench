__global__ void fma_throughput(float *out, float *scale) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  float c[16];
  for (int i = 0; i < 16; ++i) {
    c[i] = i;
  }
  for (int i = 0; i < 2048; ++i) {
    c[0] += scale[0] * c[0]; 
    c[1] += scale[1] * c[1]; 
    c[2] += scale[2] * c[2]; 
    c[3] += scale[3] * c[3]; 
    c[4] += scale[4] * c[4]; 
    c[5] += scale[5] * c[5]; 
    c[6] += scale[6] * c[6]; 
    c[7] += scale[7] * c[7]; 
    c[8] += scale[8] * c[8]; 
    c[9] += scale[9] * c[9]; 
    c[10] += scale[10] * c[10]; 
    c[11] += scale[11] * c[11]; 
    c[12] += scale[12] * c[12]; 
    c[13] += scale[13] * c[13]; 
    c[14] += scale[14] * c[14]; 
    c[15] += scale[15] * c[15]; 
  } 
  out[index] = c[0] + c[1] + c[2] + c[3]
              + c[4] + c[5] + c[6] + c[7]
              + c[8] + c[9] + c[10] + c[11]
              + c[12] + c[13] + c[14] + c[15];
}

__global__ void fma_latency(float *out, float scale) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  float c = 1.0f;
  for (int i = 0; i < 1024 * 1024 * 1024; ++i) {
    c += scale * c; 
  } 
  out[index] = c;
}
