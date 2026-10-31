"""GPU-only hot-cache CUDA Graph benchmark. All timings are measured, not targets."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import csv
import json
import statistics
import subprocess
import time
import torch
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
import lab

torch.manual_seed(20261014)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
ROWS=[]


def graph_time(fn,inner=16,groups=30):
    # Warm on a side stream, then capture identical GPU work for every implementation.
    stream=torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(10): fn()
    stream.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):
        for _ in range(inner): fn()
    for _ in range(10): graph.replay()
    torch.cuda.synchronize()
    samples=[]
    begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    for _ in range(groups):
        begin.record(); graph.replay(); end.record(); end.synchronize()
        samples.append(begin.elapsed_time(end)*1000/inner)
    return samples


def add(op,shape,variant,fn,output,reference,flops=0,workspace=0):
    fn(); torch.cuda.synchronize()
    y=output() if callable(output) else output
    if not torch.allclose(y,reference,atol=1e-2,rtol=1e-2):
        raise AssertionError((op,shape,variant,(y-reference).abs().max().item()))
    max_abs=(y.float()-reference.float()).abs().max().item()
    samples=graph_time(fn)
    median=statistics.median(samples)
    row=dict(op=op,shape=shape,variant=variant,latency_us=median,
             p10_group_mean_us=sorted(samples)[2],p90_group_mean_us=sorted(samples)[26],
             workspace_bytes=workspace,max_abs=max_abs,
             tflops=flops/(median*1e6) if flops else 0,
             timing='cuda_graph_16_calls',cache='repeated_inputs_hot',samples_us=json.dumps(samples))
    ROWS.append(row)
    print(f'{op:7} {shape:30} {variant:20} {median:10.3f} us',flush=True)


def rand(*shape): return torch.randn(shape,device='cuda',dtype=torch.float16)


def rms():
    names=['separate','fused_shared','fused_shuffle','fused_half2']
    for t,h in [(1,4096),(128,768),(128,4096),(1024,4096),(1024,8192)]:
        x,r,w=rand(t,h),rand(t,h),rand(h)
        ref=lab.rms_reference(x,r,w)[1]
        for v,name in enumerate(names):
            (_,y),fn=lab.prepare_rms(x,r,w,v)
            add('rms',f'T{t}_H{h}',name,fn,y,ref)
        u,y=torch.empty_like(x),torch.empty_like(x)
        def eager():
            torch.add(x,r,out=u)
            uf=u.float()
            y.copy_(uf*torch.rsqrt(uf.square().mean(-1,keepdim=True)+1e-6)*w.float())
        add('rms',f'T{t}_H{h}','torch_eager',eager,y,ref)


def gemm():
    for m,n,k in [(17,65,33),(1,1024,1024),(128,512,512),(512,1024,1024),(128,4096,4096)]:
        a,b=rand(m,k)*.2,rand(k,n)*.2
        ref=(a.float()@b.float()).half()
        for v,name in enumerate(['naive','shared_tile16','wmma_tile64']):
            y,fn=lab.prepare_gemm(a,b,v)
            add('gemm',f'M{m}_N{n}_K{k}',name,fn,y,ref,2*m*n*k)
        y=torch.empty((m,n),device='cuda',dtype=torch.float16)
        add('gemm',f'M{m}_N{n}_K{k}','torch_mm',lambda:torch.mm(a,b,out=y),y,ref,2*m*n*k)


def attention():
    for b,hq,hkv,l,d in [(1,32,8,128,128),(1,32,8,2048,128),(1,32,8,8192,128),(4,32,8,2048,128),(4,32,32,2048,64)]:
        q,k,v=rand(b,hq,d),rand(b,hkv,l,d),rand(b,hkv,l,d)
        lens=torch.full((b,),l,device='cuda',dtype=torch.int32)
        ref=lab.decode_reference(q,k,v,lens)
        shape=f'B{b}_Q{hq}_KV{hkv}_L{l}_D{d}'
        for variant,split,name in [(0,1,'three_stage'),(1,1,'online'),(2,2,'split2'),(2,4,'split4'),(2,8,'split8'),(2,16,'split16')]:
            y,fn=lab.prepare_decode(q,k,v,lens,variant,split)
            add('decode',shape,name,fn,y,ref,workspace=fn.workspace_bytes)
        holder={}
        def sdpa():
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                holder['y']=F.scaled_dot_product_attention(q.unsqueeze(2),k,v,is_causal=False,enable_gqa=True).squeeze(2)
        try:
            add('decode',shape,'sdpa_flash',sdpa,lambda:holder['y'],ref)
        except RuntimeError as e:
            print(f'SKIP sdpa_flash: {e}',flush=True)
            (lab.ROOT/'reports/raw/sdpa_error.txt').write_text(str(e))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--op',choices=['all','rms','gemm','decode'],default='all')
    parser.add_argument('--output',default='benchmark.csv')
    args=parser.parse_args()
    start=time.time()
    try:
        for name,fn in [('rms',rms),('gemm',gemm),('decode',attention)]:
            if args.op in ('all',name): fn()
    finally:
        dest=lab.ROOT/'reports/raw'/args.output
        if ROWS:
            with dest.open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=ROWS[0].keys());writer.writeheader();writer.writerows(ROWS)
        env=dict(timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                 gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,
                 python=sys.version,elapsed_seconds=time.time()-start,seed=20261014,
                 allow_tf32=False,allow_fp16_reduced_precision_reduction=False,
                 command=sys.argv,
                 nvidia_smi=subprocess.getoutput('nvidia-smi'))
        (lab.ROOT/'reports/raw'/f'{Path(args.output).stem}_environment.json').write_text(json.dumps(env,indent=2))
    print(f'Saved {len(ROWS)} rows to {dest}')
