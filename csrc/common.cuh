#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math_constants.h>

__device__ __forceinline__ float warp_sum(float x) {
    for (int d=16; d; d/=2) x += __shfl_down_sync(0xffffffff, x, d);
    return x;
}
__device__ __forceinline__ float warp_max(float x) {
    for (int d=16; d; d/=2) x = fmaxf(x, __shfl_down_sync(0xffffffff, x, d));
    return x;
}

// Every thread in this 256-thread block participates; all return the same sum.
template<bool SHUFFLE>
__device__ float block_sum256(float x, float* scratch) {
    int t=threadIdx.x;
    if constexpr (SHUFFLE) {
        x=warp_sum(x);
        if ((t&31)==0) scratch[t/32]=x;
        __syncthreads();
        if (t<32) {
            x=warp_sum(t<8 ? scratch[t] : 0.f);
            if (t==0) scratch[0]=x;
        }
        __syncthreads();
    } else {
        scratch[t]=x;
        __syncthreads();
        for (int d=128;d;d/=2) {
            if(t<d) scratch[t]+=scratch[t+d];
            __syncthreads();
        }
    }
    return scratch[0];
}
