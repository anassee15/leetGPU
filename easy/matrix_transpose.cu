#include <cuda_runtime.h>
#include <stdio.h>

__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int rowStride = gridDim.y * blockDim.y;
    int colStride = gridDim.x * blockDim.x;

    for (int r = row; r < rows; r += rowStride) {
        for (int c = col; c < cols; c += colStride) {
            int i = r * cols + c;
            int i_t = c * rows + r;
            output[i_t] = input[i];
        }
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int rows, int cols) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);
    matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}