"""Two shuffled-order passes for selected claims, to inspect order/clock drift."""
import sys,csv,random,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from benchmarks.run import graph_time,rand,torch,F,sdpa_kernel,SDPBackend
import lab
import statistics
cases=[]
x,r,w=rand(1024,4096),rand(1024,4096),rand(4096)
for v in (0,3):
    _,fn=lab.prepare_rms(x,r,w,v);cases.append(('rms','T1024_H4096',str(v),fn))
a,b=rand(512,1024)*.2,rand(1024,1024)*.2
for v in (0,2):
    _,fn=lab.prepare_gemm(a,b,v);cases.append(('gemm','M512_N1024_K1024',str(v),fn))
c=torch.empty((512,1024),device='cuda',dtype=torch.float16)
cases.append(('gemm','M512_N1024_K1024','torch_mm',lambda:torch.mm(a,b,out=c)))
for length in (128,8192):
    q,k,v=rand(1,32,128),rand(1,8,length,128),rand(1,8,length,128)
    lens=torch.tensor([length],device='cuda',dtype=torch.int32)
    settings=[(2,8),(2,16)] if length==128 else [(1,1),(2,16)]
    for var,split in settings:
        _,fn=lab.prepare_decode(q,k,v,lens,var,split)
        cases.append(('decode',f'B1_Q32_KV8_L{length}_D128',f'{var}_split{split}',fn))
    if length==8192:
        holder={}
        def sdpa(q=q,k=k,v=v):
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                holder['y']=F.scaled_dot_product_attention(q.unsqueeze(2),k,v,is_causal=False,enable_gqa=True)
        cases.append(('decode',f'B1_Q32_KV8_L{length}_D128','sdpa_flash',sdpa))
rows=[]
rng=random.Random(20261014)
for repeat in range(2):
    rng.shuffle(cases)
    for op,shape,variant,fn in cases:
        samples=graph_time(fn)
        value=statistics.median(samples)
        rows.append(dict(repeat=repeat,op=op,shape=shape,variant=variant,latency_us=value,samples_us=json.dumps(samples)))
        print(repeat,op,shape,variant,round(value,3),flush=True)
with (lab.ROOT/'reports/raw/confirm.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
