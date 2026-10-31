#include "api.h"
#include "rmsnorm.cuh"
#include "gemm.cuh"
#include "decode.cuh"
#include <cstdint>

extern "C" const char* lab_error(int code) { return cudaGetErrorString(static_cast<cudaError_t>(code)); }

extern "C" int lab_rms(const half* x,const half* r,const half* w,half* u,half* y,int rows,int h,float eps,int variant,cudaStream_t stream) {
    if(rows<0 || h<1 || h>8192 || eps<=0 || variant<0 || variant>3) return cudaErrorInvalidValue;
    if(rows==0) return cudaSuccess;
    if(variant==0) {
        residual_add<<<(rows*h+255)/256,256,0,stream>>>(x,r,u,rows*h);
        rms_separate<<<rows,256,0,stream>>>(u,w,y,h,eps);
    } else if(variant==1) rms_fused<false><<<rows,256,0,stream>>>(x,r,w,u,y,h,eps);
    else if(variant==3 && h%2==0 && (reinterpret_cast<uintptr_t>(x)|reinterpret_cast<uintptr_t>(r)|reinterpret_cast<uintptr_t>(w)|reinterpret_cast<uintptr_t>(u)|reinterpret_cast<uintptr_t>(y))%4==0)
        rms_half2<<<rows,256,0,stream>>>(reinterpret_cast<const half2*>(x),reinterpret_cast<const half2*>(r),reinterpret_cast<const half2*>(w),reinterpret_cast<half2*>(u),reinterpret_cast<half2*>(y),h/2,eps);
    else rms_fused<true><<<rows,256,0,stream>>>(x,r,w,u,y,h,eps);
    return cudaGetLastError();
}

extern "C" int lab_gemm(const half* a,const half* b,half* c,int m,int n,int k,int variant,cudaStream_t stream) {
    if(m<0 || n<1 || k<1 || variant<0 || variant>2) return cudaErrorInvalidValue;
    if(m==0) return cudaSuccess;
    if(variant==0) gemm_naive<<<dim3((n+15)/16,(m+15)/16),dim3(16,16),0,stream>>>(a,b,c,m,n,k);
    else if(variant==1) gemm_tiled<<<dim3((n+15)/16,(m+15)/16),dim3(16,16),0,stream>>>(a,b,c,m,n,k);
    else gemm_wmma<<<dim3((n+63)/64,(m+63)/64),128,0,stream>>>(a,b,c,m,n,k);
    return cudaGetLastError();
}

extern "C" int lab_decode(const half* q,const half* k,const half* v,const int* lengths,half* out,float* workspace,int batch,int hq,int hkv,int l,int d,int variant,int splits,cudaStream_t stream) {
    if(batch<1 || hkv<1 || hq<hkv || hq%hkv || l<1 || (d!=64 && d!=128) || variant<0 || variant>2 || splits<1 || splits>64) return cudaErrorInvalidValue;
    int heads=batch*hq;
    if(variant==0) {
        decode_qk<<<dim3((l+3)/4,heads),128,0,stream>>>(q,k,lengths,workspace,hq,hkv,l,d);
        decode_softmax<<<heads,256,0,stream>>>(workspace,lengths,workspace+heads*l,hq,l);
        decode_pv<<<heads,128,0,stream>>>(workspace+heads*l,v,lengths,out,hq,hkv,l,d);
    } else if(variant==1) decode_online<false><<<heads,128,0,stream>>>(q,k,v,lengths,out,workspace,hq,hkv,l,d,1);
    else {
        decode_online<true><<<dim3(heads,splits),128,0,stream>>>(q,k,v,lengths,out,workspace,hq,hkv,l,d,splits);
        decode_merge<<<heads,128,0,stream>>>(workspace,out,d,splits);
    }
    return cudaGetLastError();
}
