#include "api.h"
#include <vector>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <algorithm>

void ck(int e) { if(e) { fprintf(stderr,"CUDA: %s\n",lab_error(e)); std::exit(1); } }
struct Buffer {
    void* ptr=nullptr;
    explicit Buffer(size_t n) { ck(cudaMalloc(&ptr,n)); }
    ~Buffer() { cudaFree(ptr); }
    half* h() { return static_cast<half*>(ptr); }
    float* f() { return static_cast<float*>(ptr); }
    int* i() { return static_cast<int*>(ptr); }
};
std::vector<half> input(int n,int seed) {
    std::vector<half> x(n);
    for(int i=0;i<n;i++) x[i]=__float2half(std::sin(float(i*7+seed))*0.3f);
    return x;
}
void upload(Buffer& b,const std::vector<half>& x) { ck(cudaMemcpy(b.ptr,x.data(),x.size()*2,cudaMemcpyHostToDevice)); }
void verify(const char* name,Buffer& b,const std::vector<float>& ref) {
    std::vector<half> got(ref.size());
    ck(cudaMemcpy(got.data(),b.ptr,got.size()*2,cudaMemcpyDeviceToHost));
    float error=0;
    for(size_t i=0;i<got.size();i++) {
        float value=__half2float(got[i]);
        float e=std::abs(value-ref[i]); error=std::max(error,e);
        if(!std::isfinite(value) || e>0.003f+0.01f*std::abs(ref[i])) {
            fprintf(stderr,"FAIL %s i=%zu got=%g expected=%g\n",name,i,value,ref[i]); std::exit(2);
        }
    }
    printf("PASS %-12s max_abs=%g\n",name,error);
}

void rms(int rows,int h,int selected=-1,bool profile=false) {
    auto x=input(rows*h,1),r=input(rows*h,2),w=input(h,3);
    Buffer dx(x.size()*2),dr(r.size()*2),dw(h*2),du(x.size()*2),dy(x.size()*2);
    upload(dx,x);upload(dr,r);upload(dw,w);
    std::vector<float> u(x.size()),y(x.size());
    if(!profile) for(int row=0;row<rows;row++) {
        double sum=0;
        for(int j=0;j<h;j++) { int i=row*h+j;u[i]=__half2float(__float2half(__half2float(x[i])+__half2float(r[i])));sum+=u[i]*u[i]; }
        for(int j=0;j<h;j++) { int i=row*h+j;y[i]=u[i]/std::sqrt(sum/h+1e-6)*__half2float(w[j]); }
    }
    for(int v=0;v<4;v++) if(selected<0 || v==selected) {
        ck(lab_rms(dx.h(),dr.h(),dw.h(),du.h(),dy.h(),rows,h,1e-6,v,nullptr));
        ck(cudaDeviceSynchronize());
        if(!profile) {verify("rms-u",du,u);verify("rms-y",dy,y);}
    }
}

void gemm(int m,int n,int k,int selected=-1,bool profile=false) {
    auto a=input(m*k,4),b=input(k*n,5);
    Buffer da(a.size()*2),db(b.size()*2),dc(m*n*2);upload(da,a);upload(db,b);
    std::vector<float> ref(m*n);
    if(!profile) for(int i=0;i<m;i++) for(int j=0;j<n;j++) {
        double s=0;for(int z=0;z<k;z++) s+=double(__half2float(a[i*k+z]))*__half2float(b[z*n+j]);ref[i*n+j]=s;
    }
    for(int v=0;v<3;v++) if(selected<0 || v==selected) {
        ck(lab_gemm(da.h(),db.h(),dc.h(),m,n,k,v,nullptr));ck(cudaDeviceSynchronize());
        if(!profile) verify("gemm",dc,ref);
    }
}

void attention(int l,int d,int selected=-1,bool profile=false) {
    int batch=2,hq=4,hkv=2;
    auto q=input(batch*hq*d,1),k=input(batch*hkv*l*d,2),v=input(batch*hkv*l*d,3);
    Buffer dq(q.size()*2),dk(k.size()*2),dv(v.size()*2),dl(batch*4),out(q.size()*2);
    Buffer work(std::max(2*batch*hq*l,batch*hq*8*(d+2))*sizeof(float));
    upload(dq,q);upload(dk,k);upload(dv,v);
    int lens[2]={l,profile?l:0};ck(cudaMemcpy(dl.ptr,lens,sizeof(lens),cudaMemcpyHostToDevice));
    std::vector<float> ref(q.size(),0);
    if(!profile) for(int head=0;head<batch*hq;head++) {
        int b=head/hq,kh=(head%hq)/2,len=lens[b];if(!len) continue;
        std::vector<double> p(len);double denom=0,m=-1e30;
        for(int j=0;j<len;j++) { double s=0;for(int z=0;z<d;z++) s+=double(__half2float(q[head*d+z]))*__half2float(k[((b*hkv+kh)*l+j)*d+z]);p[j]=s/std::sqrt(double(d));m=std::max(m,p[j]); }
        for(auto& s:p) {s=std::exp(s-m);denom+=s;}
        for(int z=0;z<d;z++) {double s=0;for(int j=0;j<len;j++) s+=p[j]/denom*__half2float(v[((b*hkv+kh)*l+j)*d+z]);ref[head*d+z]=s;}
    }
    for(int variant=0;variant<3;variant++) if(selected<0 || variant==selected) {
        ck(lab_decode(dq.h(),dk.h(),dv.h(),dl.i(),out.h(),work.f(),batch,hq,hkv,l,d,variant,8,nullptr));ck(cudaDeviceSynchronize());
        if(!profile) verify("decode",out,ref);
    }
}

int main(int argc,char** argv) {
    if(argc==4 && std::string(argv[1])=="--profile") {
        std::string op=argv[2];int v=std::atoi(argv[3]);
        if(op=="rms") rms(1024,4096,v,true);
        else if(op=="gemm") gemm(128,512,512,v,true);
        else if(op=="decode") attention(2048,128,v,true);
        else return 3;
        return 0;
    }
    rms(3,769);rms(8,4096);gemm(17,65,33);gemm(64,128,64);attention(129,64);attention(257,128);
    puts("All native numerical checks passed.");
}
