"""Shared exact proposal engine and inference-only, rollout-local OPD Reflex.

A is a persistent fixed projector owned/saved by the EAGLE adapter. B_fast is
ONE shared compact-vocab adapter, reset every rollout. No optimizer/autograd or
model forward is used here. CUDA production requires Triton; Torch is CPU oracle.
"""
from contextlib import nullcontext
import importlib
import math
import torch

OPD_COUNTER_NAMES=(
    'opd_state_weight','opd_selected_states','opd_visited_states','opd_frontier_states',
    'opd_kl_sum','opd_union_size_sum','opd_compact_mass_sum','opd_draft_topk_target_mass_sum',
    'opd_invalid_states','opd_updates','opd_active_rows_sum','opd_rounds',
    'opd_nonfinite_kl_states',
)
GENERATION_COUNTER_NAMES=('verification_batches','active_response_rounds','verified_tree_nodes')


def initialize_projector(hidden,rank,seed=42):
    if not 1<=rank<=64:raise ValueError('OPD rank must be in [1,64]')
    # Separate generator: never consume the target sampler/global RNG.
    return torch.randn(hidden,rank,generator=torch.Generator().manual_seed(seed),dtype=torch.float32)/math.sqrt(hidden)


def select_states_reference(tree,path,visited_weight=1.,frontier_weight=1.):
    """Vectorized CPU oracle, also used for non-CUDA integration tests."""
    b,q=tree.parents.shape;rows=torch.arange(q,device=tree.parents.device)
    visited=(rows[None,:,None]==path.packed_indices[:,None,:]).any(-1)
    parent_visited=(tree.parents[:,:,None]==path.packed_indices[:,None,:]).any(-1)
    frontier=~visited&parent_visited&(rows[None,:]>0)
    expanded=tree.feedback_contexts>=0
    weights=torch.where(visited,float(visited_weight),torch.where(frontier,float(frontier_weight),0.))
    weights=torch.where(expanded,weights,0.)
    kind=torch.where((weights>0)&visited,1,torch.where((weights>0)&frontier,2,0))
    return weights,kind


def union_reference(p,q,target_ids,draft_ids):
    """Top-k union + ONE tail; not a renormalized shortlist loss."""
    k=draft_ids.shape[-1]
    ids=torch.cat((draft_ids,target_ids),-1)
    valid=torch.cat((torch.ones_like(draft_ids,dtype=torch.bool),
                     ~(target_ids[:,:,None]==draft_ids[:,None,:]).any(-1)),-1)
    pu=torch.where(valid,p.gather(-1,ids),0.);qu=torch.where(valid,q.gather(-1,ids),0.)
    empty_tail=valid.sum(-1)==p.shape[-1]
    pt=torch.where(empty_tail,0.,(1-pu.sum(-1)).clamp_min(0.))
    qt=torch.where(empty_tail,0.,(1-qu.sum(-1)).clamp_min(0.))
    kl=torch.where(pu>0,pu*(pu.log()-qu.log()),0.).sum(-1)
    kl+=torch.where(pt>0,pt*(pt.log()-qt.log()),0.)
    return ids,valid,pu,qu,pt,qt,kl


class OPDReflex:
    def __init__(self,rank=8,topk=16,fast_lr=.01,visited_weight=1.,frontier_weight=1.,
                 profile=False,diagnostics=False,enabled=True,backend='auto'):
        if not 1<=rank<=64 or topk<1 or fast_lr<0 or min(visited_weight,frontier_weight)<0:
            raise ValueError('invalid OPD rank/topk/lr/state weights')
        if backend not in ('auto','torch','triton'):raise ValueError('invalid OPD backend')
        self.rank,self.requested_topk,self.fast_lr=int(rank),int(topk),float(fast_lr)
        self.visited_weight,self.frontier_weight=float(visited_weight),float(frontier_weight)
        self.profile,self.diagnostics,self.enabled=bool(profile),bool(diagnostics),bool(enabled)
        self.requested_backend=backend;self._events=[];self._layout=None

    def start(self,model,batch,mapping,hidden_size,*,max_contexts,max_nodes,max_path,max_proposal_contexts):
        device=mapping.device;v=mapping.numel();k=min(v,self.requested_topk)
        if k<min(v,max_proposal_contexts):raise ValueError('OPD_TOPK must cover max_draft_k')
        if self.requested_backend=='torch' and device.type=='cuda':
            raise ValueError('Torch OPD is CPU oracle only; production CUDA requires Triton')
        self.backend='triton' if device.type=='cuda' else 'torch'
        if self.requested_backend=='triton' and device.type!='cuda':raise ValueError('Triton OPD requires CUDA')
        self._kernels=importlib.import_module('helper.tree_kernels') if self.backend=='triton' else None
        self._opd_kernels=importlib.import_module('helper.opd_reflex_kernels') if self.backend=='triton' else None
        self.mapping,self.vocab,self.topk,self.max_batch=mapping,v,k,batch
        self.cache_contexts=max_contexts
        layout=(batch,v,hidden_size,max_contexts,max_nodes,max_path,max_proposal_contexts,k,str(device),self.enabled)
        self.head=model.draft_model.lm_head if hasattr(model,'draft_model') else model.draft_head
        if self.enabled:
            self.projector=model.get_opd_projector(self.rank) if hasattr(model,'get_opd_projector') else None
            if self.projector is None:
                if not hasattr(model,'opd_projector'):model.opd_projector=initialize_projector(hidden_size,self.rank).to(device)
                self.projector=model.opd_projector
            if tuple(self.projector.shape)!=(hidden_size,self.rank):raise ValueError('persistent A rank/hidden mismatch')
        if layout!=self._layout:
            self._layout=layout
            def alloc(shape,dtype=torch.float32):return torch.empty(shape,device=device,dtype=dtype)
            self.bitmap=alloc((v+31)//32,torch.int32)
            self.active_count=alloc(1,torch.int32)
            self.B_fast=alloc((v,self.rank)) if self.enabled else None
            self.active_ids=alloc(v,torch.int32) if self.enabled else None
            self.proposal_u=alloc((batch*max_proposal_contexts,self.rank)) if self.enabled else None
            self.proposal_q=alloc(batch*max_proposal_contexts*k)
            self.proposal_ids=alloc(batch*max_proposal_contexts*k,torch.long)
            self.proposal_norm=alloc(batch*max_proposal_contexts*2)
            tiles=(v+255)//256+1
            self.proposal_tiles=[alloc(batch*max_proposal_contexts*tiles*(k if i>=2 else 1),torch.long if i==3 else torch.float32) for i in range(4)] if self.backend=='triton' else []
            self.path_workspace=[alloc((batch,max_path),torch.long) for _ in range(3)]+[alloc(batch,torch.long)]
            self.padded_path_workspace=[alloc((batch,max_path),torch.bool if i==2 else torch.long) for i in range(3)]+[alloc((batch,1),torch.long)]
            if self.enabled:
                self.head_cache=alloc((batch,max_contexts,hidden_size),self.head.weight.dtype)
                self.u_cache=alloc((batch,max_contexts,self.rank))
                self.ids_cache=alloc((batch,max_contexts,k),torch.long);self.q_cache=alloc((batch,max_contexts,k))
                self.norm_cache=alloc((batch,max_contexts,2))
                # max_nodes bounds TOTAL packed verification rows, not per
                # response. B*q is bounded by verification_capacity + B.
                n=max_nodes;t=(v+255)//256
                self.selected_weights=alloc(n);self.selected_kind=alloc(n,torch.int32)
                self.teacher_p=alloc(n*k);self.teacher_ids=alloc(n*k,torch.long);self.teacher_mass=alloc(n)
                self.teacher_q=alloc(n*k);self.union_ids=alloc(n*2*k,torch.long);self.union_g=alloc(n*2*k)
                self.state_stats=alloc(n*10);self.round_weight=alloc(1)
                self.teacher_tiles=[alloc(n*t*(k if i>=2 else 1),torch.long if i==3 else torch.float32) for i in range(4)] if self.backend=='triton' else []
                self.counters=alloc(len(OPD_COUNTER_NAMES),torch.float64)
        self.bitmap.zero_();self.active_count.zero_()
        if self.enabled:self.B_fast.zero_();self.counters.zero_()
        self._ever_updated=False;self._events.clear()

    def begin(self,label):
        if self.profile and self.backend=='triton':
            event=torch.cuda.Event(enable_timing=True);event.record();return label,event
        return None

    def end(self,ticket):
        if ticket is not None:
            end=torch.cuda.Event(enable_timing=True);end.record();self._events.append((ticket[0],ticket[1],end))

    @torch.no_grad()
    def propose(self,logits,hidden,k,mapping,*,root=False,context_offset=0,head_inputs=None):
        b,c,v=logits.shape;keep=self.topk
        self.logits_dtype=logits.dtype
        if k>keep or v!=self.vocab:raise ValueError('proposal k/vocabulary mismatch')
        ticket=self.begin('opd_feature_ms')
        u=None
        if self.enabled:
            if head_inputs is None:head_inputs=hidden
            u=self.proposal_u[:b*c].view(b,c,self.rank)
            if self.backend=='triton':self._opd_kernels.feature(hidden,self.projector,u,head_inputs,self,context_offset)
            else:
                with torch.autocast(device_type='cpu',enabled=False):u.copy_(hidden.float().matmul(self.projector))
        self.end(ticket);ticket=self.begin('proposal_ms')
        values=self.proposal_q[:b*c*keep].view(b,c,keep);ids=self.proposal_ids[:b*c*keep].view(b,c,keep)
        norm=self.proposal_norm[:b*c*2].view(b,c,2)
        if self.backend=='triton':
            self._opd_kernels.propose(logits,u,self,keep,self.proposal_tiles,(values,ids,norm),self.enabled and self._ever_updated)
        else:
            z=logits.float().clone()
            if self.enabled and self._ever_updated:
                active=self.active_ids[:int(self.active_count)]
                z[...,active.long()]+=u.matmul(self.B_fast[active.long()].t())
            maximum=z.amax(-1);total=(z-maximum[...,None]).exp().sum(-1)
            norm[...,0].copy_(maximum);norm[...,1].copy_(total)
            # Deterministic low-ID tie convention, same for OFF and OPD.
            order=torch.argsort(z,dim=-1,descending=True,stable=True)[...,:keep]
            probabilities=z.softmax(-1)
            ids.copy_(order);values.copy_(probabilities.gather(-1,order))
        if self.enabled:
            if head_inputs is None:
                if hasattr(self.head,'opd_inputs'):head_inputs=self.head.opd_inputs(hidden)
                else:head_inputs=hidden
            if self.backend=='triton':self._opd_kernels.cache_proposal(values,ids,norm,self,context_offset)
            else:
                self.head_cache[:b,context_offset:context_offset+c].copy_(head_inputs)
                self.u_cache[:b,context_offset:context_offset+c].copy_(u)
                self.ids_cache[:b,context_offset:context_offset+c].copy_(ids)
                self.q_cache[:b,context_offset:context_offset+c].copy_(values)
                self.norm_cache[:b,context_offset:context_offset+c].copy_(norm)
        self.end(ticket)
        if self.backend=='torch':
            # CPU oracle only: reproduce legacy Torch's K-dependent tie rule
            # for integration parity. CUDA production NEVER takes this path:
            # it uses one Top16 scan and its shared low-ID tie convention.
            tree_q,tree_ids=torch.topk(probabilities,k=k,dim=-1)
            return tree_q,tree_ids,mapping[tree_ids]
        return values[...,:k],ids[...,:k],mapping[ids[...,:k]]

    @torch.no_grad()
    def feedback(self,tree,path,target,*,greedy=False):
        if not self.enabled:return
        if self.backend=='triton':self._opd_kernels.feedback(self,tree,path,target,greedy)
        else:self._feedback_reference(tree,path,target,greedy)
        if self.fast_lr>0:self._ever_updated=True

    def _feedback_reference(self,tree,path,target,greedy):
        # CPU oracle only: dense reconstructed q and gradient permitted HERE.
        b,q=tree.parents.shape;k=self.topk
        weights,kind=select_states_reference(tree,path,self.visited_weight,self.frontier_weight)
        batch=torch.arange(b)[:,None].expand(b,q);context=tree.feedback_contexts.clamp_min(0)
        head_inputs=self.head_cache[batch,context];u=self.u_cache[batch,context]
        raw=torch.nn.functional.linear(head_inputs,self.head.weight,self.head.bias).float()
        corrected=raw+u.matmul(self.B_fast.t())
        norm=self.norm_cache[batch,context]
        draft=((corrected-norm[...,0,None]).exp()/norm[...,1,None])
        if greedy:teacher=(self.mapping[None,None,:]==target[...,None]).float()
        else:teacher=target[...,self.mapping].float()
        mass=teacher.sum(-1);good=torch.isfinite(mass)&(mass>0)
        teacher=teacher/torch.where(good,mass,1.)[...,None]
        w=torch.where(good,weights,0.).reshape(-1)
        di=self.ids_cache[batch,context].reshape(-1,k)
        ti=torch.argsort(teacher,dim=-1,descending=True,stable=True)[...,:k].reshape(-1,k)
        ids,valid,p,qq,pt,qt,kl=union_reference(teacher.reshape(-1,self.vocab),draft.reshape(-1,self.vocab),ti,di)
        # Cached top-k q avoids selected-row output-head reconstruction drift.
        qq[:,:k]=self.q_cache[batch,context].reshape(-1,k)
        qt=torch.where(valid.sum(-1)==self.vocab,0.,(1-qq.sum(-1)).clamp_min(0.))
        kl=torch.where(p>0,p*(p.log()-qq.log()),0.).sum(-1)+torch.where(pt>0,pt*(pt.log()-qt.log()),0.)
        selected=w>0;g=torch.where(valid,(qq-p)*w[:,None],0.)
        total=w.sum()
        if self.fast_lr>0 and total>0:
            delta=torch.zeros_like(self.B_fast)
            delta.index_add_(0,ids.flatten(),(g[...,None]*u.reshape(-1,self.rank)[:,None,:]).reshape(-1,self.rank))
            self.B_fast.add_(delta,alpha=-self.fast_lr/float(total))
            active=torch.nonzero(self.B_fast.abs().sum(-1)>0,as_tuple=False).flatten()
            self.active_count.fill_(active.numel());self.active_ids[:active.numel()].copy_(active)
        count=selected.sum();categories=kind.flatten()
        finite=torch.isfinite(kl)
        stats=torch.stack((total,count,(selected&(categories==1)).sum(),(selected&(categories==2)).sum(),
            torch.where(selected&finite,w*kl,0.).sum(),(valid&selected[:,None]).sum(),
            torch.where(selected,mass.flatten(),0.).sum(),torch.where(selected,p[:,:k].sum(-1),0.).sum(),
            ((weights>0)&~good).sum(),((total>0)&(self.fast_lr>0)).float(),self.active_count[0].float(),total.new_tensor(1.),
            (selected&~finite).sum()))
        self.counters.add_(stats.to(torch.float64))

    def remove_finished(self,indices):
        # B is shared, not row-owned. Current contexts are overwritten next tree.
        pass

    def finish(self):
        if self.enabled:
            payload=self.counters
            if self.diagnostics:
                extra=torch.stack((self.B_fast.norm().double(),self.B_fast.abs().amax().double(),self.active_count[0].double()))
                payload=torch.cat((payload,extra))
            packet=payload.cpu().tolist()  # exactly ONE counter/diagnostic packet
        else:packet=[0.]*len(OPD_COUNTER_NAMES)
        result=dict(zip(OPD_COUNTER_NAMES,packet[:len(OPD_COUNTER_NAMES)]))
        if self.enabled and self.diagnostics:
            result.update(dict(zip(('opd_final_b_norm','opd_final_b_max_abs','opd_final_active_rows'),packet[len(OPD_COUNTER_NAMES):])))
        sections={}
        if self.profile:
            for label,start,end in self._events:
                if not end.query():end.synchronize()  # opt-in profiling after rollout only
                sections[label]=sections.get(label,0.)+start.elapsed_time(end)
            sections.setdefault('opd_proposal_extra_ms',0.)
        result['opd_profile_sections_ms']=sections if self.profile else None
        result['opd_backend']=self.backend if self.enabled else 'off'
        # Full proposal is an auxiliary inclusive timer, not an additional OPD
        # phase. Wait can overlap OPD-stream work. Neither is double-counted.
        result['opd_profile_time_ms']=sum(v for k,v in sections.items() if k not in ('opd_wait_ms','proposal_ms'))
        self._events.clear()
        return result

    def clear(self):
        # Runtime pools are deliberately retained by model cache. No per-round
        # teacher or batch hidden references survive the feedback call.
        self._events.clear()
        if self.enabled:self.B_fast.zero_();self.bitmap.zero_();self.active_count.zero_()
        self._ever_updated=False
