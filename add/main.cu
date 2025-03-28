#include <iostream>
#include <cuda.h>
#include <cuda_fp16.h>
#include "cuda_runtime.h"
#include <cmath>
using namespace std;
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])
#define CEIL(a,b) ((a+b-1)/(b))
#define cudaCheck(err) _cudaCheck(err, __FILE__, __LINE__)
void _cudaCheck(cudaError_t error, const char *file, int line){
    if (error != cudaSuccess){
        printf("[CUDA ERROR] at file %s(line %d):\n%s \n", file, line, cudaGetErrorString(error));
    }
    return;
}


__global__ void add(float *a, float *b, float *out, int N){
    int idx = (blockDim.x * blockIdx.x + threadIdx.x) * 4;
    if(idx > N)
        return;

    float4 tmp_a = FLOAT4(a[idx]);
    float4 tmp_b = FLOAT4(a[idx]);
    float4 tmp_c;
    tmp_c.x = tmp_a.x + tmp_b.x;
    tmp_c.y = tmp_a.y + tmp_b.y;
    tmp_c.z = tmp_a.z + tmp_b.z;
    tmp_c.w = tmp_a.w + tmp_b.w;
    FLOAT4(out[idx]) = tmp_c;

}


int main() {

    std::cout << "Hello, World!" << std::endl;
    return 0;
}
