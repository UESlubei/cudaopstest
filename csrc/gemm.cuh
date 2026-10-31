#pragma once
#include "common.cuh"
#include <mma.h>

__global__ void gemm_naive(const half* a,const half* b,half* c,int m,int n,int k) {
    int col=blockIdx.x*16+threadIdx.x, row=blockIdx.y*16+threadIdx.y;
    if(row>=m || col>=n) return;
    float acc=0;
    for(int z=0;z<k;z++) acc+=__half2float(a[row*k+z])*__half2float(b[z*n+col]);
    c[row*n+col]=__float2half(acc);
}

__global__ void gemm_tiled(const half* a,const half* b,half* c,int m,int n,int k) {
    __shared__ float as[16][16], bs[16][16];
    int x=threadIdx.x,y=threadIdx.y,col=blockIdx.x*16+x,row=blockIdx.y*16+y;
    float acc=0;
    for(int base=0;base<k;base+=16) {
        as[y][x]=(row<m && base+x<k) ? __half2float(a[row*k+base+x]) : 0.f;
        bs[y][x]=(base+y<k && col<n) ? __half2float(b[(base+y)*n+col]) : 0.f;
        __syncthreads();
        #pragma unroll
        for(int z=0;z<16;z++) acc+=as[y][z]*bs[z][x];
        __syncthreads();
    }
    if(row<m && col<n) c[row*n+col]=__float2half(acc);
}

// 4 warps cover one 64x64 block; each warp computes four 16x16 fragments.
// Staging into aligned shared memory makes arbitrary input tails safe for WMMA.
__global__ void gemm_wmma(const half* a,const half* b,half* c,int m,int n,int k) {
    using namespace nvcuda;
    __shared__ __align__(32) half as[64*16], bs[16*64];
    __shared__ __align__(32) float out[64*64];
    int t=threadIdx.x, warp=t/32, wr=warp/2, wc=warp%2;
    int r0=blockIdx.y*64,c0=blockIdx.x*64;
    wmma::fragment<wmma::accumulator,16,16,16,float> acc[2][2];
    #pragma unroll
    for(int i=0;i<2;i++) for(int j=0;j<2;j++) wmma::fill_fragment(acc[i][j],0.f);
    for(int base=0;base<k;base+=16) {
        for(int p=t;p<1024;p+=128) {
            int ar=p/16,ac=p%16,br=p/64,bc=p%64;
            as[p]=(r0+ar<m && base+ac<k) ? a[(r0+ar)*k+base+ac] : __float2half(0);
            bs[p]=(base+br<k && c0+bc<n) ? b[(base+br)*n+c0+bc] : __float2half(0);
        }
        __syncthreads();
        wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> af[2];
        wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::row_major> bf[2];
        #pragma unroll
        for(int i=0;i<2;i++) {
            wmma::load_matrix_sync(af[i],as+(wr*32+i*16)*16,16);
            wmma::load_matrix_sync(bf[i],bs+wc*32+i*16,64);
        }
        #pragma unroll
        for(int i=0;i<2;i++) for(int j=0;j<2;j++) wmma::mma_sync(acc[i][j],af[i],bf[j],acc[i][j]);
        __syncthreads();
    }
    #pragma unroll
    for(int i=0;i<2;i++) for(int j=0;j<2;j++)
        wmma::store_matrix_sync(out+(wr*32+i*16)*64+wc*32+j*16,acc[i][j],64,wmma::mem_row_major);
    __syncthreads();
    for(int p=t;p<4096;p+=128) {
        int r=r0+p/64,col=c0+p%64;
        if(r<m && col<n) c[r*n+col]=__float2half(out[p]);
    }
}
