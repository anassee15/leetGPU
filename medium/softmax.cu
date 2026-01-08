#include <cuda_runtime.h>

__global__ void online_softmax(const float* input, float* output, int N) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    float local_max = -INFINITY;
    float sum_e = 0.0f;

    for(int i=0; i < N; i += 1) {
        if (input[i] > local_max) {
            sum_e *= __expf(local_max - input[i]);
            local_max = input[i];
        }
        sum_e += __expf(input[i] - local_max);
    }

    for(int i=tid; i < N; i+= stride)
        output[i] = __expf(input[i] - local_max) / sum_e;
    
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    online_softmax<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
