"""Small inference-only CUDA binding. Kernels use PyTorch's current stream.

prepare_* validates and allocates once, returning (outputs, launch).
Keep the closure alive: it owns tensors and workspace used by asynchronous kernels.
"""
from pathlib import Path
import ctypes as C
import math
import torch

ROOT = Path(__file__).resolve().parent
lib = C.CDLL(str(ROOT / 'build/libllm_cuda_lab.so'))
P, I, F = C.c_void_p, C.c_int, C.c_float
lib.lab_rms.argtypes = [P]*5 + [I,I,F,I,P]
lib.lab_gemm.argtypes = [P]*3 + [I]*4 + [P]
lib.lab_decode.argtypes = [P]*6 + [I]*7 + [P]
for name in ('lab_rms','lab_gemm','lab_decode'):
    getattr(lib, name).restype = I
lib.lab_error.argtypes = [I]
lib.lab_error.restype = C.c_char_p


def check(code):
    if code:
        raise RuntimeError(lib.lab_error(code).decode())


def tensors_ok(*xs):
    if not xs or any(x.device.type != 'cuda' or x.dtype != torch.float16 or not x.is_contiguous() for x in xs):
        raise ValueError('expected contiguous CUDA float16 tensors')
    if any(x.device != xs[0].device for x in xs):
        raise ValueError('all tensors must be on the same device')
    if any(x.requires_grad for x in xs):
        raise ValueError('inference-only: requires_grad is unsupported')
    if any(x.numel() >= 2**31 for x in xs):
        raise ValueError('32-bit indexing limit exceeded')


def launch_on_device(device, call):
    with torch.cuda.device(device):
        check(call(torch.cuda.current_stream(device).cuda_stream))


def prepare_rms(x, r, w, variant=3, eps=1e-6):
    tensors_ok(x,r,w)
    if x.ndim != 2 or r.shape != x.shape or w.shape != (x.shape[1],):
        raise ValueError('expected X/R [T,H], weight [H]')
    rows,h=x.shape
    if not 1<=h<=8192 or variant not in range(4) or not math.isfinite(eps) or eps<=0:
        raise ValueError('H must be 1..8192, variant 0..3, eps finite and positive')
    u,y=torch.empty_like(x),torch.empty_like(x)
    def run():
        launch_on_device(x.device, lambda stream: lib.lab_rms(x.data_ptr(),r.data_ptr(),w.data_ptr(),u.data_ptr(),y.data_ptr(),rows,h,eps,variant,stream))
    return (u,y),run


def prepare_gemm(a,b,variant=2):
    tensors_ok(a,b)
    if a.ndim!=2 or b.ndim!=2 or a.shape[1]!=b.shape[0] or b.shape[0]<1 or b.shape[1]<1 or variant not in range(3):
        raise ValueError('expected A[M,K] and B[K,N], K,N>0, variant 0..2')
    m,k=a.shape; n=b.shape[1]
    if m*n>=2**31 or (m+15)//16>65535:
        raise ValueError('shape exceeds supported index/grid limits')
    c=torch.empty((m,n),device=a.device,dtype=a.dtype)
    def run():
        launch_on_device(a.device, lambda stream: lib.lab_gemm(a.data_ptr(),b.data_ptr(),c.data_ptr(),m,n,k,variant,stream))
    return c,run


def prepare_decode(q,k,v,lengths,variant=2,splits=8):
    tensors_ok(q,k,v)
    if q.ndim!=3 or k.ndim!=4 or v.shape!=k.shape or q.shape[0]!=k.shape[0] or q.shape[-1]!=k.shape[-1]:
        raise ValueError('expected Q[B,Hq,D], K/V[B,Hkv,L,D]')
    batch,hq,d=q.shape; _,hkv,l,_=k.shape
    if batch<1 or hkv<1 or hq<hkv or hq%hkv or l<1 or d not in (64,128) or variant not in range(3) or not 1<=splits<=64:
        raise ValueError('invalid attention configuration')
    if batch*hq>65535 or batch*hq*max(l,splits*(d+2))>=2**31:
        raise ValueError('shape exceeds supported index/grid limits')
    if lengths.device!=q.device or lengths.dtype!=torch.int32 or lengths.shape!=(batch,) or not lengths.is_contiguous():
        raise ValueError('lengths must be contiguous CUDA int32 [B]')
    # One-time synchronized validation is outside steady-state timing/capture.
    # lengths is immutable for the returned launcher; recreate it to change lengths.
    if bool(((lengths<0)|(lengths>l)).any().item()):
        raise ValueError('lengths out of range')
    size=2*batch*hq*l if variant==0 else batch*hq*splits*(d+2) if variant==2 else 1
    workspace=torch.empty(size,device=q.device,dtype=torch.float32)
    out=torch.empty_like(q)
    def run():
        launch_on_device(q.device,lambda stream: lib.lab_decode(q.data_ptr(),k.data_ptr(),v.data_ptr(),lengths.data_ptr(),out.data_ptr(),workspace.data_ptr(),batch,hq,hkv,l,d,variant,splits,stream))
    run.workspace_bytes=workspace.numel()*workspace.element_size()
    return out,run


def rms_reference(x,r,w,eps=1e-6):
    u=(x.float()+r.float()).half()
    uf=u.float()
    return u,(uf*torch.rsqrt(uf.square().mean(-1,keepdim=True)+eps)*w.float()).half()


def decode_reference(q,k,v,lengths):
    out=torch.zeros_like(q)
    g=q.shape[1]//k.shape[1]
    for b,length in enumerate(lengths.cpu().tolist()):
        if length:
            kb=k[b,:,:length].repeat_interleave(g,dim=0).float()
            vb=v[b,:,:length].repeat_interleave(g,dim=0).float()
            score=(q[b].float().unsqueeze(1)*kb).sum(-1)/math.sqrt(q.shape[-1])
            out[b]=(score.softmax(-1).unsqueeze(-1)*vb).sum(1).half()
    return out
