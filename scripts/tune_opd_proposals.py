#!/usr/bin/env python3
"""Offline exact sparse/dense crossover measurement on the ACTUAL GPU.

No model/transformer forward or training. Produces a fingerprinted profile;
profiles from RTX3090 are NOT B200 profiles. No guessed crossover at runtime.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

def measure(fn,iterations,torch):
    for _ in range(5):fn()
    torch.cuda.synchronize()
    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);samples=[]
    for _ in range(iterations):
        a.record();fn();b.record();b.synchronize();samples.append(a.elapsed_time(b))
    return statistics.median(samples)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True);p.add_argument('--shapes',default='64x1,64x8')
    p.add_argument('--vocab',type=int,default=32768);p.add_argument('--hidden',type=int,default=2048)
    p.add_argument('--rank',type=int,default=8);p.add_argument('--slots',default='0,16,128,512,2048,8192,32768')
    p.add_argument('--iterations',type=int,default=30);p.add_argument('--dtype',choices=['bf16','fp32'],default='bf16')
    a=p.parse_args();output=Path(a.output)
    if output.exists():p.error('choose a new output profile')
    import torch
    import numpy as np
    from types import SimpleNamespace
    from helper.opd_reflex import OPDReflex,initialize_projector
    from helper import opd_reflex_kernels as kernels
    if not torch.cuda.is_available():p.error('real CUDA needed; no fake timings')
    torch.manual_seed(42);dtype=torch.bfloat16 if a.dtype=='bf16' else torch.float32
    thresholds={};records=[]
    for shape in a.shapes.split(','):
        b,c=map(int,shape.split('x'));mapping=torch.arange(a.vocab,device='cuda')
        head=torch.nn.Linear(a.hidden,a.vocab,bias=False,dtype=dtype,device='cuda')
        model=SimpleNamespace(draft_head=head,opd_projector=initialize_projector(a.hidden,a.rank).cuda())
        s=OPDReflex(a.rank,16,backend='triton')
        s.start(model,b,mapping,a.hidden,max_contexts=c,max_nodes=b*c,max_path=6,max_proposal_contexts=c)
        raw=torch.randn(b,c,a.vocab,device='cuda',dtype=dtype);h=torch.randn(b,c,a.hidden,device='cuda',dtype=dtype)
        trials=[]
        for count in sorted(set(min(a.vocab,int(x)) for x in a.slots.split(','))):
            s.B_fast.zero_();s.bitmap.zero_();s.active_count.fill_(count)
            ids=torch.randperm(a.vocab,device='cuda')[:count].sort().values
            s.active_ids[:count]=ids;s.B_fast[ids]=torch.randn(count,a.rank,device='cuda')*.05
            # Outside measurement/hot path, seeding via CPU is deliberate.
            cpu_ids=ids.cpu().numpy()
            bits=np.zeros((a.vocab+31)//32,dtype=np.uint32)
            np.bitwise_or.at(bits,cpu_ids//32,np.left_shift(np.uint32(1),(cpu_ids%32).astype(np.uint32)))
            s.bitmap.copy_(torch.from_numpy(bits.view(np.int32)));s._ever_updated=bool(count)
            results={};times={}
            for mode in ('sparse','dense'):
                s.proposal_mode=mode
                q,i,_=s.propose(raw,h,8,mapping);results[mode]=(q.clone(),i.clone())
                times[mode]=measure(lambda:s.propose(raw,h,8,mapping),a.iterations,torch)
            if not all(torch.equal(x,y) for x,y in zip(results['sparse'],results['dense'])):
                raise AssertionError('sparse/dense BITWISE probability/ID parity failed; no profile')
            trials.append(dict(slots=count,**times,bitwise_parity=True))
        candidates=sorted({x['slots'] for x in trials}|{a.vocab+1})
        selected=min(candidates,key=lambda t:sum(x['sparse'] if x['slots']<t else x['dense'] for x in trials))
        key=f'{b},{c},{a.vocab},{a.rank},{dtype}';thresholds[key]=selected
        records.append(dict(shape=shape,threshold=selected,trials=trials))
    payload=dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,triton=kernels.triton.__version__,
        cuda=torch.version.cuda,kernel_sha256=hashlib.sha256(Path(kernels.__file__).read_bytes()).hexdigest(),
        thresholds=thresholds,records=records,
        dense='tiled FP32 rank GEMM, canonical ordered adds, no FMA/TF32 drift',
        note='Component crossover only. Validate end-to-end B200 sweep before adoption. Untuned shapes remain sparse.')
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps(payload,indent=2))

if __name__=='__main__':main()
