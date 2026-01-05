#include <cuda_runtime.h>


__device__ __forceinline__ float atomicMaxFloat (float * addr, float value) {
    float old;
    old = (value >= 0) ? __int_as_float(atomicMax((int *)addr, __float_as_int(value))) :
         __uint_as_float(atomicMin((unsigned int *)addr, __float_as_uint(value)));

    return old;
}

__global__ void mymax(const float* input, float* output, int N) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    double value = -INFINITY;
    for (int i=tid; i < N; i += stride) {
        value = fmaxf(value, input[i]);
    }

    extern __shared__ double s[];
    s[threadIdx.x] = value;
    __syncthreads();

    for(int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset)
            s[threadIdx.x] = fmaxf(s[threadIdx.x], s[threadIdx.x + offset]);
        __syncthreads();
    }

    if (threadIdx.x == 0)
        atomicMaxFloat(output, (float)s[0]);
}

__global__ void mysum(const float* input, float* output, float* max_v, int N) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    double value = 0.0;
    for (int i=tid; i < N; i += stride) {
        value += __expf(input[i]-*max_v);
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

__global__ void softmax_kernel(const float* input, float* output, float* max_v, float* sum_e, int N) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for(int i=tid; i < N; i+= stride)
        output[i] = __expf(input[i] - *max_v) / *sum_e;
    
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    size_t smem = threadsPerBlock * sizeof(double);

    float* sum_e;
    float* max_v;

    cudaMalloc(&sum_e, sizeof(float));
    cudaMemset(sum_e, 0, sizeof(float));

    cudaMalloc(&max_v, sizeof(float));
    cudaMemset(max_v, 0, sizeof(float));

    mymax<<<blocksPerGrid, threadsPerBlock, smem>>>(input, max_v, N);
    cudaDeviceSynchronize();

    mysum<<<blocksPerGrid, threadsPerBlock, smem>>>(input, sum_e, max_v, N);
    cudaDeviceSynchronize();

    softmax_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, max_v, sum_e, N);
    cudaDeviceSynchronize();

    cudaFree(sum_e);
    cudaFree(max_v);
}
