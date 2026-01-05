#include <cuda_runtime.h>

__global__ void softmax_kernel(const float* input, float* output, int N) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    // naive implementation first

    if (tid < N) {
        // get the max
        float max_value = input[0];
        for(int i=1; i<N; i+=1) 
            max_value = fmaxf(max_value, input[i]);

        // get the sum 
        float sum_e = 0;
        for(int i=0; i<N; i+=1) 
            sum_e += __expf(input[i] - max_value);

        for(int i=tid; i < N; i+= stride)
            output[tid] = __expf(input[i] - max_value) / sum_e;
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    softmax_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
