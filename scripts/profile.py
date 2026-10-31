"""Capture comparable representative workloads, outside benchmark timing."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel,SDPBackend
import lab
torch.manual_seed(20261014)
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
def rand(*s): return torch.randn(s,device='cuda',dtype=torch.float16)
work=[]
x,r,w=rand(1024,4096),rand(1024,4096),rand(4096)
for v in range(4):
    out,fn=lab.prepare_rms(x,r,w,v);work.append((f'rms_{v}',fn))
a,b=rand(512,1024),rand(1024,1024)
for v in range(3):
    out,fn=lab.prepare_gemm(a,b,v);work.append((f'gemm_{v}',fn))
q,k,v=rand(1,32,128),rand(1,8,8192,128),rand(1,8,8192,128)
lens=torch.tensor([8192],device='cuda',dtype=torch.int32)
for var,split in [(0,1),(1,1),(2,16)]:
    out,fn=lab.prepare_decode(q,k,v,lens,var,split);work.append((f'decode_{var}_split{split}',fn))
holder={}
def sdpa():
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        holder['out']=F.scaled_dot_product_attention(q.unsqueeze(2),k,v,is_causal=False,enable_gqa=True)
work.append(('sdpa_flash',sdpa))
for _,fn in work:
    for _ in range(5): fn()
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart()
for name,fn in work:
    torch.cuda.nvtx.range_push(name)
    fn()
    torch.cuda.nvtx.range_pop()
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
print('Captured 11 operator calls; timings here are profiling-only.')
