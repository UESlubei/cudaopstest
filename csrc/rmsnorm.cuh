#pragma once
#include "common.cuh"

__global__ void residual_add(const half* x,const half* r,half* u,int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) u[i]=__float2half(__half2float(x[i])+__half2float(r[i]));
}

__global__ void rms_separate(const half* u,const half* w,half* y,int h,float eps) {
    __shared__ float scratch[256];
    int t=threadIdx.x, row=blockIdx.x*h;
    float sum=0;
    for(int j=t;j<h;j+=256) { float v=__half2float(u[row+j]); sum+=v*v; }
    float inv=rsqrtf(block_sum256<false>(sum,scratch)/h+eps);
    for(int j=t;j<h;j+=256) y[row+j]=__float2half(__half2float(u[row+j])*inv*__half2float(w[j]));
}

// Unrolled bounded storage makes the retained row values visible to the compiler.
// H<=8192; larger widths are rejected at the Python/API boundary.
template<bool SHUFFLE>
__global__ void rms_fused(const half* x,const half* r,const half* w,half* u,half* y,int h,float eps) {
    __shared__ float scratch[256];
    float vals[32];
    float sum=0;
    int row=blockIdx.x*h, t=threadIdx.x;
    #pragma unroll
    for(int i=0;i<32;i++) {
        int j=t+i*256;
        float v=0;
        if(j<h) {
            half v16=__float2half(__half2float(x[row+j])+__half2float(r[row+j]));
            u[row+j]=v16;
            v=__half2float(v16);
        }
        vals[i]=v; sum+=v*v;
    }
    float inv=rsqrtf(block_sum256<SHUFFLE>(sum,scratch)/h+eps);
    #pragma unroll
    for(int i=0;i<32;i++) {
        int j=t+i*256;
        if(j<h) y[row+j]=__float2half(vals[i]*inv*__half2float(w[j]));
    }
}

__global__ void rms_half2(const half2* x,const half2* r,const half2* w,half2* u,half2* y,int pairs,float eps) {
    __shared__ float scratch[256];
    float2 vals[16];
    float sum=0;
    int row=blockIdx.x*pairs, t=threadIdx.x;
    #pragma unroll
    for(int i=0;i<16;i++) {
        int j=t+i*256;
        float2 v=make_float2(0,0);
        if(j<pairs) {
            half2 v16=__hadd2(x[row+j],r[row+j]);
            u[row+j]=v16; v=__half22float2(v16);
        }
        vals[i]=v; sum+=v.x*v.x+v.y*v.y;
    }
    float inv=rsqrtf(block_sum256<true>(sum,scratch)/(2*pairs)+eps);
    #pragma unroll
    for(int i=0;i<16;i++) {
        int j=t+i*256;
        if(j<pairs) {
            float2 v=vals[i], g=__half22float2(w[j]);
            y[row+j]=__floats2half2_rn(v.x*inv*g.x,v.y*inv*g.y);
        }
    }
}
