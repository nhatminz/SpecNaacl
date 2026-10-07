#!/usr/bin/env python3
"""Offline exact sparse/dense crossover measurement on the ACTUAL GPU.

No model/transformer forward or training. Produces a fingerprinted profile;
profiles from RTX3090 are NOT B200 profiles. Includes fused and workspace dense
paths, auto dispatch, exact parity, normalized workload interpolation.
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
    p.add_argument('--output',required=True);p.add_argument('--shapes',default='1x1,8x1,32x1,64x1,16x8,32x8,64x8')
    p.add_argument('--draft-config');p.add_argument('--vocab',type=int);p.add_argument('--hidden',type=int)
    p.add_argument('--rank',type=int,default=8);p.add_argument('--slots',default='0,16,64,256,1024,4096,8192,V')
    p.add_argument('--iterations',type=int,default=30);p.add_argument('--dtype',choices=['bf16','fp16','fp32'],default='bf16')
    a=p.parse_args();output=Path(a.output)
    if a.draft_config:
        config=json.loads(Path(a.draft_config).read_text())
        vocab=int(config.get('draft_vocab_size') or config['vocab_size']);hidden=int(config['hidden_size'])
        if a.vocab is not None and a.vocab!=vocab:p.error('explicit vocab differs from production draft config')
        if a.hidden is not None and a.hidden!=hidden:p.error('explicit hidden differs from production draft config')
        a.vocab,a.hidden=vocab,hidden
    if a.vocab is None or a.hidden is None:p.error('--draft-config required, or explicit --vocab AND --hidden for synthetic kernel tests')
    if output.exists():p.error('choose a new output profile')
    import torch
    import numpy as np
    from types import SimpleNamespace
    from helper.opd_reflex import OPDReflex,initialize_projector
    from helper import opd_reflex_kernels as kernels
    if not torch.cuda.is_available():p.error('real CUDA needed; no fake timings')
    torch.manual_seed(42);dtype={'bf16':torch.bfloat16,'fp16':torch.float16,'fp32':torch.float32}[a.dtype]
    thresholds={};dense_implementations={};records=[]
    for shape in a.shapes.split(','):
        b,c=map(int,shape.split('x'));mapping=torch.arange(a.vocab,device='cuda')
        head=torch.nn.Linear(a.hidden,a.vocab,bias=False,dtype=dtype,device='cuda')
        model=SimpleNamespace(draft_head=head,opd_projector=initialize_projector(a.hidden,a.rank).cuda())
        s=OPDReflex(a.rank,16,backend='triton')
        s.start(model,b,mapping,a.hidden,max_contexts=c,max_nodes=b*c,max_path=6,max_proposal_contexts=c)
        raw=torch.randn(b,c,a.vocab,device='cuda',dtype=dtype);h=torch.randn(b,c,a.hidden,device='cuda',dtype=dtype)
        trials=[]
        for count in sorted(set(min(a.vocab,a.vocab if x=='V' else int(x)) for x in a.slots.split(','))):
            s.B_fast.zero_();s.bitmap.zero_();s.active_count.fill_(count)
            ids=torch.randperm(a.vocab,device='cuda')[:count].sort().values
            s.active_ids[:count]=ids;s.B_fast[ids]=torch.randn(count,a.rank,device='cuda')*.05
            # Outside measurement/hot path, seeding via CPU is deliberate.
            cpu_ids=ids.cpu().numpy()
            bits=np.zeros((a.vocab+31)//32,dtype=np.uint32)
            np.bitwise_or.at(bits,cpu_ids//32,np.left_shift(np.uint32(1),(cpu_ids%32).astype(np.uint32)))
            s.bitmap.copy_(torch.from_numpy(bits.view(np.int32)));s._ever_updated=bool(count)
            s.host_active_count=count
            results={};times={}
            for mode in ('sparse','dense','dense_gemm','auto'):
                s.proposal_mode='dense' if mode=='dense_gemm' else mode
                s.dense_implementation='gemm' if mode=='dense_gemm' else 'fused'
                q,i,_=s.propose(raw,h,16,mapping);results[mode]=(q.clone(),i.clone(),s.proposal_norm[:b*c*2].clone())
                times[mode]=measure(lambda:s.propose(raw,h,8,mapping),a.iterations,torch)
            for mode in ('dense','dense_gemm','auto'):
                if not all(torch.equal(x,y) for x,y in zip(results['sparse'],results[mode])):
                    details=[(torch.count_nonzero(x!=y).item(),(x-y).abs().max().item()) for x,y in zip(results['sparse'],results[mode])]
                    raise AssertionError(f'shape={shape} S={count}: sparse/{mode} BITWISE probability/ID/normalizer parity failed {details}; no profile')
            trials.append(dict(slots=count,**times,bitwise_parity=True))
        candidates=sorted({x['slots'] for x in trials}|{a.vocab+1})
        dense_key=min(('dense','dense_gemm'),key=lambda mode:sum(x[mode] for x in trials if x['slots']>0))
        selected=min(candidates,key=lambda t:sum(x['sparse'] if x['slots']<t else x[dense_key] for x in trials if x['slots']>0))
        key=f'{b},{c},{a.vocab},{a.rank},{dtype}';thresholds[key]=selected
        dense_implementations[key]='gemm' if dense_key=='dense_gemm' else 'fused'
        records.append(dict(shape=shape,contexts=b*c,threshold=selected,dense_implementation=dense_implementations[key],trials=trials))
        del s,model,head,raw,h,results
    # Auto timings above intentionally measure the uncalibrated default.
    payload=dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,triton=kernels.triton.__version__,
        cuda=torch.version.cuda,kernel_sha256=hashlib.sha256(Path(kernels.__file__).read_bytes()).hexdigest(),
        thresholds=thresholds,dense_implementations=dense_implementations,records=records,
        dense='fused rank-8 correction + normalization + Top16; dense_gemm separately measured',
        note='Component crossover only; validate end-to-end. Auto timings use uncalibrated V/8; profile selects measured dense implementation and crossover, interpolated in log(contexts). No B200 claim on other GPUs.')
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps(payload,indent=2))

if __name__=='__main__':main()
