#include <cuda_runtime.h>
#include <stdio.h>

__global__ void invert_kernel(unsigned char* image, int width, int height) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int idx = tid; idx < (width * height); idx += stride) {
        for (auto i = 0; i < 3; i++)
            image[idx * 4 + i] = 255 - image[idx * 4 + i];
    }
}
// image_input, image_output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(unsigned char* image, int width, int height) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (width * height + threadsPerBlock - 1) / threadsPerBlock;

    invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(image, width, height);
    cudaDeviceSynchronize();
}