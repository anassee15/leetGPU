#include <cuda_runtime.h>

__global__ void sum(const float* input, float* output, int N) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    double value = 0.0;
    for (int i=tid; i < N; i += stride) {
        value += input[i];
    }

    extern __shared__ double s[];
    s[threadIdx.x] = value;
    __syncthreads();

    for(int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset)
            s[threadIdx.x] += s[threadIdx.x + offset];
        __syncthreads();
    }

    if (threadIdx.x == 0)
        atomicAdd(output, (float)s[0]);
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int threadPerBlock = 256;
    int numBlocks = 256;
    size_t smem = threadPerBlock * sizeof(double);
    cudaMemset(output, 0, sizeof(float)); // Add this before the kernel launch

    sum<<<numBlocks, threadPerBlock, smem>>>(input, output, N);
    cudaDeviceSynchronize();
}