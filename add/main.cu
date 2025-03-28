#include <iostream>
#include <cuda.h>
#include <cuda_fp16.h>
#include "cuda_runtime.h"
#include <cmath>
#include <vector>
#include <cassert>
using namespace std;

#define CEIL(a, b) ((a + b - 1) / (b))
#define cudaCheck(err) _cudaCheck(err, __FILE__, __LINE__)

void _cudaCheck(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess) {
        printf("[CUDA ERROR] at file %s(line %d):\n%s \n", file, line, cudaGetErrorString(error));
    }
    return;
}

// Define a structure for int8_4 to handle int8 data in a similar way to float4
struct int8_4 {
    int8_t x, y, z, w;
};

// Define a structure for int32_4 to handle int32 data in a similar way to float4
struct int32_4 {
    int32_t x, y, z, w;
};

// Define a structure for half4 to handle fp16 data in a similar way to float4
struct half4 {
    half x, y, z, w;
};

// Template for the add kernel
template <typename T>
__global__ void add(T *a, T *b, T *out, int N) {
    int idx = (blockDim.x * blockIdx.x + threadIdx.x) * 4;
    if (idx >= N)
        return;

    // Load data into registers
    T tmp_a = *reinterpret_cast<T*>(&a[idx]);
    T tmp_b = *reinterpret_cast<T*>(&b[idx]);
    T tmp_c;

    // Perform element-wise addition
    tmp_c.x = tmp_a.x + tmp_b.x;
    tmp_c.y = tmp_a.y + tmp_b.y;
    tmp_c.z = tmp_a.z + tmp_b.z;
    tmp_c.w = tmp_a.w + tmp_b.w;

    // Store the result
    *reinterpret_cast<T*>(&out[idx]) = tmp_c;
}

// Specialization for float
template <>
__global__ void add<float>(float *a, float *b, float *out, int N) {
    int idx = (blockDim.x * blockIdx.x + threadIdx.x) * 4;
    if (idx >= N)
        return;

    float4 tmp_a = *reinterpret_cast<float4*>(&a[idx]);
    float4 tmp_b = *reinterpret_cast<float4*>(&b[idx]);
    float4 tmp_c;

    tmp_c.x = tmp_a.x + tmp_b.x;
    tmp_c.y = tmp_a.y + tmp_b.y;
    tmp_c.z = tmp_a.z + tmp_b.z;
    tmp_c.w = tmp_a.w + tmp_b.w;

    *reinterpret_cast<float4*>(&out[idx]) = tmp_c;
}

// Specialization for half
template <>
__global__ void add<half>(half *a, half *b, half *out, int N) {
    int idx = (blockDim.x * blockIdx.x + threadIdx.x) * 4;
    if (idx >= N)
        return;

    half4 tmp_a = *reinterpret_cast<half4*>(&a[idx]);
    half4 tmp_b = *reinterpret_cast<half4*>(&b[idx]);
    half4 tmp_c;

    tmp_c.x = __hadd(tmp_a.x, tmp_b.x);
    tmp_c.y = __hadd(tmp_a.y, tmp_b.y);
    tmp_c.z = __hadd(tmp_a.z, tmp_b.z);
    tmp_c.w = __hadd(tmp_a.w, tmp_b.w);

    *reinterpret_cast<half4*>(&out[idx]) = tmp_c;
}

// Specialization for int8_t
template <>
__global__ void add<int8_t>(int8_t *a, int8_t *b, int8_t *out, int N) {
    int idx = (blockDim.x * blockIdx.x + threadIdx.x) * 4;
    if (idx >= N)
        return;

    int8_4 tmp_a = *reinterpret_cast<int8_4*>(&a[idx]);
    int8_4 tmp_b = *reinterpret_cast<int8_4*>(&b[idx]);
    int8_4 tmp_c;

    tmp_c.x = tmp_a.x + tmp_b.x;
    tmp_c.y = tmp_a.y + tmp_b.y;
    tmp_c.z = tmp_a.z + tmp_b.z;
    tmp_c.w = tmp_a.w + tmp_b.w;

    *reinterpret_cast<int8_4*>(&out[idx]) = tmp_c;
}

// Specialization for int32_t
template <>
__global__ void add<int32_t>(int32_t *a, int32_t *b, int32_t *out, int N) {
    int idx = (blockDim.x * blockIdx.x + threadIdx.x) * 4;
    if (idx >= N)
        return;

    int32_4 tmp_a = *reinterpret_cast<int32_4*>(&a[idx]);
    int32_4 tmp_b = *reinterpret_cast<int32_4*>(&b[idx]);
    int32_4 tmp_c;

    tmp_c.x = tmp_a.x + tmp_b.x;
    tmp_c.y = tmp_a.y + tmp_b.y;
    tmp_c.z = tmp_a.z + tmp_b.z;
    tmp_c.w = tmp_a.w + tmp_b.w;

    *reinterpret_cast<int32_4*>(&out[idx]) = tmp_c;
}

// CPU add function
template <typename T>
void addCPU(T* a, T* b, T* out, int N) {
    for (int i = 0; i < N; ++i) {
        out[i] = a[i] + b[i];
    }
}

// Function to compare results
template <typename T>
void compareResults(T* cudaResult, T* cpuResult, int N) {
    bool success = true;
    for (int i = 0; i < N; ++i) {
        if (cudaResult[i] != cpuResult[i]) {
            success = false;
            break;
        }
    }
    if (success) {
        std::cout << "Results match!" << std::endl;
    } else {
        std::cout << "Results do not match!" << std::endl;
    }
}

template <typename T>
void startTest() {
    std::cout << "Start!" << std::endl;
    int N = 1024;
    T *h_a, *h_b, *h_out_cuda, *h_out_cpu;

    h_a = new T[N];
    h_b = new T[N];
    h_out_cuda = new T[N];
    h_out_cpu = new T[N];

    T *d_a, *d_b, *d_out;
    cudaMalloc((void**)&d_a, N * sizeof(T));
    cudaMalloc((void**)&d_b, N * sizeof(T));
    cudaMalloc((void**)&d_out, N * sizeof(T));

    for (int i = 0; i < N; i++) {
        h_a[i] = static_cast<T>(i);
        h_b[i] = static_cast<T>(i);
    }

    cudaMemcpy(d_a, h_a, N * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(T), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = CEIL(CEIL(N, 4), blockSize);
    add<T><<<numBlocks, blockSize>>>(d_a, d_b, d_out, N);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());

    cudaMemcpy(h_out_cuda, d_out, N * sizeof(T), cudaMemcpyDeviceToHost);

    // Perform CPU computation
    addCPU(h_a, h_b, h_out_cpu, N);

    // Compare results
    compareResults(h_out_cuda, h_out_cpu, N);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    delete[] h_a;
    delete[] h_b;
    delete[] h_out_cuda;
    delete[] h_out_cpu;
}

int main() {
    startTest<float>();
    startTest<half>();
    startTest<int8_t>();
    startTest<int32_t>();

    return 0;
}