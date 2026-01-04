#include <cuda_runtime.h>

__global__ void clip_kernel(const float* input, float* output, float lo, float hi, int N) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = gridDim.x * blockDim.x;

    for(int i=tid; i < N; i += stride) {
        float value = input[i];
        output[i] = value > hi ? hi : value < lo ? lo : value;
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, float lo, float hi, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    clip_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, lo, hi, N);
    cudaDeviceSynchronize();
}
