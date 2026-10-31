#pragma once
#include "common.cuh"

// Baseline: materialize FP32 scores and probabilities in two separate arrays.
__global__ void decode_qk(const half* q,const half* k,const int* lengths,float* scores,int hq,int hkv,int l,int d) {
    int head=blockIdx.y, batch=head/hq, kh=(head%hq)/(hq/hkv);
    int token=blockIdx.x*4+threadIdx.x/32,lane=threadIdx.x%32;
    if(token>=l) return; // whole warp exits together
    float s=0;
    if(token<lengths[batch]) {
        for(int z=lane;z<d;z+=32) s+=__half2float(q[head*d+z])*__half2float(k[((batch*hkv+kh)*l+token)*d+z]);
        s=warp_sum(s);
    } else s=-CUDART_INF_F;
    if(lane==0) scores[head*l+token]=s*rsqrtf(float(d));
}

__global__ void decode_softmax(const float* scores,const int* lengths,float* p,int hq,int l) {
    __shared__ float scratch[256], maximum;
    int head=blockIdx.x,t=threadIdx.x,len=lengths[head/hq];
    float m=-CUDART_INF_F;
    for(int j=t;j<len;j+=256) m=fmaxf(m,scores[head*l+j]);
    scratch[t]=m; __syncthreads();
    for(int s=128;s;s/=2) { if(t<s) scratch[t]=fmaxf(scratch[t],scratch[t+s]); __syncthreads(); }
    if(t==0) maximum=len ? scratch[0] : 0.f;
    __syncthreads();
    float sum=0;
    for(int j=t;j<len;j+=256) sum+=expf(scores[head*l+j]-maximum);
    float denom=block_sum256<true>(sum,scratch);
    for(int j=t;j<l;j+=256) p[head*l+j]=j<len ? expf(scores[head*l+j]-maximum)/denom : 0.f;
}

__global__ void decode_pv(const float* p,const half* v,const int* lengths,half* out,int hq,int hkv,int l,int d) {
    int head=blockIdx.x,batch=head/hq,kh=(head%hq)/(hq/hkv),z=threadIdx.x;
    if(z>=d) return;
    float sum=0;
    for(int j=0;j<lengths[batch];j++) sum+=p[head*l+j]*__half2float(v[((batch*hkv+kh)*l+j)*d+z]);
    out[head*d+z]=__float2half(sum);
}

template<bool SPLIT>
__global__ void decode_online(const half* q,const half* k,const half* v,const int* lengths,half* out,float* state,int hq,int hkv,int l,int d,int splits) {
    __shared__ float prob[32], running_m, running_l, rescale;
    int head=blockIdx.x,batch=head/hq,kh=(head%hq)/(hq/hkv);
    int t=threadIdx.x,lane=t%32,warp=t/32;
    int len=lengths[batch],part=blockIdx.y,chunk=(len+splits-1)/splits;
    int start=part*chunk,end=min(start+chunk,len);
    if(t==0) { running_m=-CUDART_INF_F; running_l=0.f; }
    __syncthreads();
    float acc=0;
    for(int base=start;base<end;base+=32) {
        // Each warp computes eight Q.K dot products; 32 scores per tile.
        for(int j=warp;j<32;j+=4) {
            int token=base+j;
            float s=0;
            if(token<end) {
                for(int z=lane;z<d;z+=32) s+=__half2float(q[head*d+z])*__half2float(k[((batch*hkv+kh)*l+token)*d+z]);
                s=warp_sum(s);
            } else s=-CUDART_INF_F;
            if(lane==0) prob[j]=s*rsqrtf(float(d));
        }
        __syncthreads();
        if(warp==0) {
            float score=prob[lane];
            float tile_m=warp_max(score);
            tile_m=__shfl_sync(0xffffffff,tile_m,0);
            float new_m=fmaxf(running_m,tile_m);
            float p=expf(score-new_m), tile_l=warp_sum(p);
            if(lane==0) {
                rescale=expf(running_m-new_m);
                running_l=rescale*running_l+tile_l;
                running_m=new_m;
            }
            prob[lane]=p;
        }
        __syncthreads();
        if(t<d) {
            acc*=rescale;
            for(int j=0;j<32 && base+j<end;j++)
                acc+=prob[j]*__half2float(v[((batch*hkv+kh)*l+base+j)*d+t]);
        }
        __syncthreads(); // all consumers finish before the next score tile
    }
    if constexpr(SPLIT) {
        int idx=(head*splits+part)*(d+2);
        if(t<d) state[idx+t]=acc;
        if(t==0) { state[idx+d]=running_m; state[idx+d+1]=running_l; }
    } else if(t<d) out[head*d+t]=__float2half(running_l>0 ? acc/running_l : 0.f);
}

__global__ void decode_merge(const float* state,half* out,int d,int splits) {
    int head=blockIdx.x,t=threadIdx.x;
    if(t>=d) return;
    float m=-CUDART_INF_F;
    for(int s=0;s<splits;s++) {
        int idx=(head*splits+s)*(d+2);
        if(state[idx+d+1]>0) m=fmaxf(m,state[idx+d]);
    }
    float denom=0,acc=0;
    for(int s=0;s<splits;s++) {
        int idx=(head*splits+s)*(d+2);
        if(state[idx+d+1]>0) {
            float scale=expf(state[idx+d]-m);
            denom+=scale*state[idx+d+1]; acc+=scale*state[idx+t];
        }
    }
    out[head*d+t]=__float2half(denom>0 ? acc/denom : 0.f);
}
