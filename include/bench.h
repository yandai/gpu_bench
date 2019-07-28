__global__ void shared_memory_latency(int *out, const int stride);
__global__ void shared_memory_throughput(int *out, const int stride);

// FMA
__global__ void fma_throughput(float *out, float *scale);
__global__ void fma_latency(float *out, float scale);

__global__ void init_ptr_chase(size_t *p, int buf_size, int stride);

__global__ void ldr_to_use_latency(int *out, size_t *in);
