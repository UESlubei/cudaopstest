"""Executable numerical/contract tests; no pytest dependency required."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import torch
import lab

torch.manual_seed(20261014)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
results=[]


def compare(name,got,ref,atol,rtol):
    diff=(got.float()-ref.float())
    max_abs=diff.abs().max().item() if diff.numel() else 0.
    rel=(diff.norm()/ref.float().norm().clamp_min(1e-12)).item()
    passed=bool(torch.isfinite(got).all() and torch.allclose(got,ref,atol=atol,rtol=rtol))
    results.append(dict(name=name,max_abs=max_abs,relative_l2=rel,passed=passed))
    if not passed:
        raise AssertionError(results[-1])


def expect_error(name,fn):
    try: fn()
    except ValueError:
        results.append(dict(name=name,passed=True)); return
    raise AssertionError(f'{name}: expected ValueError')


def half(*shape):
    return torch.randn(shape,device='cuda',dtype=torch.float16)


def run():
    for rows,h in [(0,33),(1,1),(3,33),(7,769),(8,768),(2,4097),(32,4096),(3,8192)]:
        for scale in (0.,1.,20.):
            x,r,w=half(rows,h)*scale,half(rows,h)*scale,half(h)
            ur,yr=lab.rms_reference(x,r,w)
            for v in range(4):
                (u,y),fn=lab.prepare_rms(x,r,w,v); fn()
                compare(f'rms-u:{rows},{h},s{scale},v{v}',u,ur,0,0)
                compare(f'rms-y:{rows},{h},s{scale},v{v}',y,yr,2e-3,2e-3)
    # Contiguous but 2-byte-offset storage: half2 path must safely fall back.
    x=half(2*768+1)[1:].view(2,768); r=half(2,768); w=half(768)
    (u,y),fn=lab.prepare_rms(x,r,w,3); fn()
    compare('rms-unaligned',y,lab.rms_reference(x,r,w)[1],2e-3,2e-3)
    for m,n,k in [(0,65,33),(1,1,1),(17,65,33),(127,255,129),(64,128,64),(128,256,512)]:
        for scale in (0.,0.2,1.):
            a,b=half(m,k)*scale,half(k,n)*scale
            ref=(a.double()@b.double()).half()
            for v in range(3):
                out,fn=lab.prepare_gemm(a,b,v); fn()
                compare(f'gemm:{m},{n},{k},s{scale},v{v}',out,ref,1e-2,1e-2)
    for batch,hq,hkv,l,d in [(1,4,4,1,64),(2,4,2,127,64),(2,8,2,129,128),(2,4,1,257,64),(1,8,2,2048,128)]:
        for scale in (0.,1.,5.):
            q,k,v=half(batch,hq,d)*scale,half(batch,hkv,l,d)*scale,half(batch,hkv,l,d)
            for lens in ([l]*batch, [0]+[max(1,l//2)]*(batch-1)):
                lengths=torch.tensor(lens,device='cuda',dtype=torch.int32)
                ref=lab.decode_reference(q,k,v,lengths)
                for variant,split in [(0,1),(1,1),(2,1),(2,4),(2,8),(2,64)]:
                    out,fn=lab.prepare_decode(q,k,v,lengths,variant,split); fn()
                    compare(f'decode:{batch},{hq},{hkv},{l},{d},s{scale},len{lens},v{variant},split{split}',out,ref,1e-2,1e-2)
    x=half(2,64); w=half(64)
    expect_error('noncontiguous',lambda:lab.prepare_rms(x.t(),x.t(),half(2)))
    expect_error('wrong-dtype',lambda:lab.prepare_rms(x.float(),x.float(),w))
    expect_error('invalid-H',lambda:lab.prepare_rms(half(1,8193),half(1,8193),half(8193)))
    expect_error('wrong-gemm-shape',lambda:lab.prepare_gemm(half(3,4),half(5,6)))
    q,k,v=half(1,4,64),half(1,2,32,64),half(1,2,32,64)
    expect_error('invalid-length',lambda:lab.prepare_decode(q,k,v,torch.tensor([33],device='cuda',dtype=torch.int32)))
    expect_error('invalid-gqa',lambda:lab.prepare_decode(half(1,3,64),k,v,torch.tensor([32],device='cuda',dtype=torch.int32)))
    expect_error('requires-grad',lambda:lab.prepare_rms(x.requires_grad_(),x,w))
    # Producer and every kernel run on a non-default stream, followed by event ordering.
    stream=torch.cuda.Stream()
    with torch.cuda.stream(stream):
        x,r,w=half(7,769),half(7,769),half(769)
        (_,y),fn=lab.prepare_rms(x,r,w); fn(); ref=lab.rms_reference(x,r,w)[1]
        a,b=half(17,33),half(33,65); c,fg=lab.prepare_gemm(a,b); fg(); cr=(a.float()@b.float()).half()
        q,k,v=half(1,4,64),half(1,2,129,64),half(1,2,129,64)
        lens=torch.tensor([127],device='cuda',dtype=torch.int32)
        o,fa=lab.prepare_decode(q,k,v,lens); fa(); ar=lab.decode_reference(q,k,v,lens)
        done=torch.cuda.Event(); done.record()
    torch.cuda.current_stream().wait_event(done)
    compare('stream-rms',y,ref,2e-3,2e-3)
    compare('stream-gemm',c,cr,1e-2,1e-2)
    compare('stream-decode',o,ar,1e-2,1e-2)


if __name__=='__main__':
    try:
        run()
    finally:
        path=lab.ROOT/'reports/raw/correctness.json'
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(results,indent=2))
    print(f'PASS: {len(results)} checks; results in {path}')
