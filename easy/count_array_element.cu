#include <cuda_runtime.h>

__global__ void count_equal_kernel(const int* input, int* output, int N, int K) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int local = 0;

    for(int i=tid; i < N; i += stride)
        local = input[i] == K ? local + 1 : local;
    
    // Reduction within block
    __shared__ int s[256];
    s[threadIdx.x] = local;
    __syncthreads();

    // Reduce in shared memory
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) s[threadIdx.x] += s[threadIdx.x + offset];
        __syncthreads();
    }

    // Atomic add to global output
    if (threadIdx.x == 0)
        atomicAdd(output, s[0]);
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output, int N, int K) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    count_equal_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N, K);
    cudaDeviceSynchronize();
}