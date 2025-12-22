#include <cuda_runtime.h>

__global__ void count_2d_equal_kernel(const int* input, int* output, int N, int M, int K) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    int strideCol = blockDim.x * gridDim.x;
    int strideRow = blockDim.y * gridDim.y;

    int tid = row * M + col;  
    int tid_local = threadIdx.y * blockDim.x + threadIdx.x;

    int local = 0;

    for (int r = row; r < N; r += strideRow) {
        for (int c = col; c < M; c += strideCol) {
            int idx = r * M + c;
            local += (input[idx] == K);
        }
    }

    // Reduction within block
    __shared__ int s[16*16];
    s[tid_local] = local;
    __syncthreads();

    // Reduce in shared memory
    for(int offset = (blockDim.x*blockDim.y) / 2; offset > 0; offset >>= 1)
    {
        if (tid_local < offset)
            s[tid_local] += s[tid_local + offset];
        __syncthreads();
    }

    // Atomic add to global output
    if (tid_local == 0) 
        atomicAdd(output, s[0]);
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output, int N, int M, int K) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((M + threadsPerBlock.x - 1) / threadsPerBlock.x,
                              (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    count_2d_equal_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N, M, K);
    cudaDeviceSynchronize();
}