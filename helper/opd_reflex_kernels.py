"""Inference-only OPD kernels. Shared B, compact vocabulary, no model/autograd/RNG.

Proposal scans inactive raw logits and ONLY S active adapter rows, then merges
exact global top-k. No [contexts,V,r] correction or probability cache exists.
Feedback reads the existing post-sampling teacher probabilities once in the
compact domain; never another target softmax/sort or transformer forward.
"""
import torch
import triton
import triton.language as tl
from helper.tree_kernels import _proposal_merge


@triton.jit
def _feature(H,A,U,HEAD_IN,HC,UC,HS0,HS1,HS2,NS0,NS1,NS2,C:tl.constexpr,HIDDEN:tl.constexpr,R:tl.constexpr,
             CACHE:tl.constexpr,OFFSET:tl.constexpr,BH:tl.constexpr,BR:tl.constexpr):
    row=tl.program_id(0).to(tl.int64)
    h,r=tl.arange(0,BH),tl.arange(0,BR)
    x=tl.load(H+row//C*HS0+row%C*HS1+h*HS2,h<HIDDEN,other=0).to(tl.float32)
    a=tl.load(A+h[:,None]*R+r[None,:],(h[:,None]<HIDDEN)&(r[None,:]<R),other=0)
    projected=tl.sum(x[:,None]*a,axis=0)
    cached=row//C*CACHE+OFFSET+row%C
    tl.store(U+row*R+r,projected,r<R);tl.store(UC+cached*R+r,projected,r<R)
    inputs=tl.load(HEAD_IN+row//C*NS0+row%C*NS1+h*NS2,h<HIDDEN,other=0)
    tl.store(HC+cached*HIDDEN+h,inputs,h<HIDDEN)


def feature(hidden,projector,out,head_inputs,state,offset):
    b,c,h=hidden.shape;r=projector.shape[1]
    _feature[(b*c,)](hidden,projector,out,head_inputs,state.head_cache,state.u_cache,
        *hidden.stride(),*head_inputs.stride(),c,h,r,state.cache_contexts,offset,
        triton.next_power_of_2(h),triton.next_power_of_2(r),num_warps=8,enable_fp_fusion=False)


@triton.jit(do_not_specialize=['ENABLED'])
def _raw_scan(Z,BITS,MAX,SUM,VALUES,IDS,ZS0,ZS1,ZS2,C:tl.constexpr,V:tl.constexpr,
              K:tl.constexpr,TILES:tl.constexpr,BV:tl.constexpr,ENABLED):
    tile,context,batch=tl.program_id(0),tl.program_id(1),tl.program_id(2).to(tl.int64)
    v=tile*BV+tl.arange(0,BV)
    z=tl.load(Z+batch*ZS0+context*ZS1+v*ZS2,v<V,other=0).to(tl.float32)
    live=v<V
    if ENABLED:
        bits=tl.load(BITS+v//32,v<V,other=0)
        live=live&(((bits>>(v%32))&1)==0)
    z=tl.where(live,z,-float('inf'))
    maximum=tl.max(z,axis=0)
    total=tl.sum(tl.where(live,tl.exp(z-tl.where(maximum>-float('inf'),maximum,0.)),0.),axis=0)
    offset=(batch*C+context)*TILES+tile
    tl.store(MAX+offset,maximum);tl.store(SUM+offset,total)
    if tile==0:
        tail=(batch*C+context)*TILES+TILES-1
        local=tl.arange(0,BV)
        tl.store(MAX+tail,-float('inf'));tl.store(SUM+tail,0.)
        tl.store(VALUES+tail*K+local,-float('inf'),local<K)
        tl.store(IDS+tail*K+local,V,local<K)
    for j in range(K):
        score=tl.max(z,axis=0)
        token=tl.min(tl.where((z==score)&live,v,V),axis=0)
        tl.store(VALUES+offset*K+j,score);tl.store(IDS+offset*K+j,token)
        z=tl.where(v==token,-float('inf'),z);live=live&(v!=token)


@triton.jit
def _active_scan(Z,U,B,ACTIVE,COUNT,MAX,SUM,VALUES,IDS,ZS0,ZS1,ZS2,
                 C:tl.constexpr,V:tl.constexpr,R:tl.constexpr,K:tl.constexpr,
                 TILES:tl.constexpr,BS:tl.constexpr,BR:tl.constexpr):
    context,batch=tl.program_id(0),tl.program_id(1).to(tl.int64)
    count=tl.load(COUNT)
    s,r=tl.arange(0,BS),tl.arange(0,BR)
    u=tl.load(U+(batch*C+context)*R+r,r<R,other=0)
    best=tl.full((BS,),-float('inf'),tl.float32);best_ids=tl.full((BS,),V,tl.int32)
    maximum=tl.full((),-float('inf'),tl.float32);total=tl.full((),0.,tl.float32)
    # Runtime loop S: no inactive token loads B[r] or performs an r-dot.
    for start in range(0,count,BS):
        token=tl.load(ACTIVE+start+s,start+s<count,other=0).to(tl.int64)
        weight=tl.load(B+token[:,None]*R+r[None,:],(start+s[:,None]<count)&(r[None,:]<R),other=0)
        raw=tl.load(Z+batch*ZS0+context*ZS1+token*ZS2,start+s<count,other=0).to(tl.float32)
        score=tl.where(start+s<count,raw+tl.sum(weight*u[None,:],axis=1),-float('inf'))
        m=tl.max(score,axis=0);new_max=tl.maximum(maximum,m)
        total=total*tl.exp(maximum-new_max)+tl.sum(tl.exp(score-new_max),axis=0)
        maximum=new_max
        combined=tl.join(best,score).reshape((2*BS,));ids=tl.join(best_ids,token.to(tl.int32)).reshape((2*BS,))
        valid=tl.join(s<K,start+s<count).reshape((2*BS,))
        best=tl.full((BS,),-float('inf'),tl.float32);best_ids=tl.full((BS,),V,tl.int32)
        for j in range(K):
            value=tl.max(combined,axis=0)
            selected=tl.min(tl.where((combined==value)&valid,ids,V),axis=0)
            best=tl.where(s==j,value,best);best_ids=tl.where(s==j,selected,best_ids)
            combined=tl.where(ids==selected,-float('inf'),combined);valid=valid&(ids!=selected)
    offset=(batch*C+context)*TILES+TILES-1
    tl.store(MAX+offset,maximum);tl.store(SUM+offset,total)
    tl.store(VALUES+offset*K+s,best,s<K);tl.store(IDS+offset*K+s,best_ids,s<K)


def propose(logits,u,state,k,workspace,outputs,enabled):
    b,c,v=logits.shape;tiles=triton.cdiv(v,256)+1
    maxima,sums,values,ids=[p[:b*c*tiles*(k if i>=2 else 1)] for i,p in enumerate(workspace)]
    out,indices,norm=outputs
    _raw_scan[(tiles-1,c,b)](logits,state.bitmap,maxima,sums,values,ids,*logits.stride(),
        c,v,k,tiles,256,enabled,num_warps=4,enable_fp_fusion=False)
    if enabled:
        ticket=state.begin('opd_proposal_extra_ms')
        _active_scan[(c,b)](logits,u,state.B_fast,state.active_ids,state.active_count,
            maxima,sums,values,ids,*logits.stride(),c,v,state.rank,k,tiles,128,
            triton.next_power_of_2(state.rank),num_warps=4,enable_fp_fusion=False)
        state.end(ticket)
    _proposal_merge[(b*c,)](maxima,sums,values,ids,out,indices,norm,c,v,k,tiles,
        triton.next_power_of_2(tiles),triton.next_power_of_2(tiles*k),num_warps=4,enable_fp_fusion=False)


@triton.jit
def _cache_proposal(Q,IDS,NORM,QC,IC,NC,C:tl.constexpr,CACHE:tl.constexpr,
                    OFFSET:tl.constexpr,K:tl.constexpr,BK:tl.constexpr):
    row=tl.program_id(0).to(tl.int64);k=tl.arange(0,BK);n=tl.arange(0,2)
    cached=row//C*CACHE+OFFSET+row%C
    tl.store(QC+cached*K+k,tl.load(Q+row*K+k,k<K,other=0),k<K)
    tl.store(IC+cached*K+k,tl.load(IDS+row*K+k,k<K,other=0),k<K)
    tl.store(NC+cached*2+n,tl.load(NORM+row*2+n))


def cache_proposal(q,ids,norm,state,offset):
    b,c,k=q.shape
    _cache_proposal[(b*c,)](q,ids,norm,state.q_cache,state.ids_cache,state.norm_cache,
        c,state.cache_contexts,offset,k,triton.next_power_of_2(k),num_warps=4)


@triton.jit
def _select(PARENTS,CONTEXTS,PATH,WEIGHTS,KIND,ROWS:tl.constexpr,WIDTH:tl.constexpr,
            PS0,PS1,VW:tl.constexpr,FW:tl.constexpr,BR:tl.constexpr,BW:tl.constexpr):
    batch=tl.program_id(0).to(tl.int64)
    row,pathslot=tl.arange(0,BR),tl.arange(0,BW)
    visited=tl.load(PATH+batch*PS0+pathslot*PS1,pathslot<WIDTH,other=-2)
    parent=tl.load(PARENTS+batch*ROWS+row,row<ROWS,other=-2)
    context=tl.load(CONTEXTS+batch*ROWS+row,row<ROWS,other=-1)
    on_path=tl.sum((row[:,None]==visited[None,:]).to(tl.int32),axis=1)>0
    parent_visited=tl.sum((parent[:,None]==visited[None,:]).to(tl.int32),axis=1)>0
    frontier=~on_path&parent_visited&(row>0)
    kind=tl.where(on_path,1,tl.where(frontier,2,0))
    weight=tl.where(on_path,VW,tl.where(frontier,FW,0.))
    valid=(row<ROWS)&(context>=0)&(weight>0)
    tl.store(WEIGHTS+batch*ROWS+row,tl.where(valid,weight,0.),row<ROWS)
    tl.store(KIND+batch*ROWS+row,tl.where(valid,kind,0),row<ROWS)


def select_states(tree,path,weights,kind,visited_weight,frontier_weight):
    b,q=tree.parents.shape
    _select[(b,)](tree.parents,tree.feedback_contexts,path.packed_indices,weights,kind,q,path.packed_indices.shape[1],
        *path.packed_indices.stride(),visited_weight,frontier_weight,triton.next_power_of_2(q),
        triton.next_power_of_2(path.packed_indices.shape[1]),num_warps=4)


@triton.jit
def _teacher_scan(TARGET,MAP,W,MAX,SUM,VALUES,IDS,TS0,TS1,TS2,
                  ROWS:tl.constexpr,V:tl.constexpr,K:tl.constexpr,TILES:tl.constexpr,
                  BV:tl.constexpr,GREEDY:tl.constexpr):
    tile,state=tl.program_id(0),tl.program_id(1).to(tl.int64)
    v=tile*BV+tl.arange(0,BV)
    selected=tl.load(W+state)>0
    if selected:
        target_id=tl.load(MAP+v,v<V,other=0)
        if GREEDY:
            p=((target_id==tl.load(TARGET+state//ROWS*TS0+state%ROWS*TS1))&(v<V)).to(tl.float32)
        else:
            p=tl.load(TARGET+state//ROWS*TS0+state%ROWS*TS1+target_id*TS2,v<V,other=0).to(tl.float32)
    else:p=tl.full((BV,),0.,tl.float32)
    offset=state*TILES+tile
    tl.store(MAX+offset,0.);tl.store(SUM+offset,tl.sum(p,axis=0))
    score=tl.where(v<V,p,-float('inf'));live=v<V
    for j in range(K):
        value=tl.max(score,axis=0);token=tl.min(tl.where((score==value)&live,v,V),axis=0)
        tl.store(VALUES+offset*K+j,value);tl.store(IDS+offset*K+j,token)
        score=tl.where(v==token,-float('inf'),score);live=live&(v!=token)


def teacher(target,mapping,weights,topk,pools,outputs,greedy=False):
    b,q=weights.shape;v=mapping.numel();tiles=triton.cdiv(v,256);n=b*q;k=min(topk,v)
    maxima,sums,values,ids=[p[:n*tiles*(k if i>=2 else 1)] for i,p in enumerate(pools)]
    probs,indices,norm=outputs
    _teacher_scan[(tiles,n)](target,mapping,weights,maxima,sums,values,ids,
        target.stride(0),target.stride(1),0 if greedy else target.stride(2),q,v,k,tiles,256,greedy,num_warps=4)
    # _proposal_merge exponentiates score values; teacher top-k instead needs
    # literal probability ranking and division by compact mass.
    _teacher_merge[(n,)](sums,values,ids,probs,indices,norm,n,v,k,tiles,
        triton.next_power_of_2(tiles),triton.next_power_of_2(tiles*k),num_warps=4)


@triton.jit
def _teacher_merge(SUM,VALUES,IDS,P,OUT_IDS,MASS,N:tl.constexpr,V:tl.constexpr,K:tl.constexpr,
                   TILES:tl.constexpr,BT:tl.constexpr,BK:tl.constexpr):
    state=tl.program_id(0).to(tl.int64);t=tl.arange(0,BT);candidate=tl.arange(0,BK)
    mass=tl.sum(tl.load(SUM+state*TILES+t,t<TILES,other=0),axis=0)
    valid_mass=(mass>0)&(mass<float('inf'))
    tl.store(MASS+state,mass)
    value=tl.load(VALUES+state*TILES*K+candidate,candidate<TILES*K,other=-float('inf'))
    ids=tl.load(IDS+state*TILES*K+candidate,candidate<TILES*K,other=V)
    for j in range(K):
        best=tl.max(value,axis=0);token=tl.min(tl.where(value==best,ids,V),axis=0)
        tl.store(P+state*K+j,tl.where(valid_mass,tl.div_rn(best,tl.where(valid_mass,mass,1.)),0.))
        tl.store(OUT_IDS+state*K+j,token)
        value=tl.where(ids==token,-float('inf'),value)


@triton.jit
def _selected_head(H,HEAD,BIAS,U,B,D_IDS,D_Q,T_IDS,NORM,W,MASS,CONTEXTS,OUT,
                   HEAD_DTYPE:tl.constexpr,ROWS:tl.constexpr,CACHE:tl.constexpr,
                   HIDDEN:tl.constexpr,R:tl.constexpr,K:tl.constexpr,
                   HAS_BIAS:tl.constexpr,BH:tl.constexpr,BR:tl.constexpr,BK:tl.constexpr):
    j,state=tl.program_id(0),tl.program_id(1).to(tl.int64)
    weight=tl.load(W+state);context=tl.load(CONTEXTS+state)
    token=tl.load(T_IDS+state*K+j)
    k=tl.arange(0,BK);cached_ids=tl.load(D_IDS+(state//ROWS*CACHE+tl.maximum(context,0))*K+k,k<K,other=-1)
    matches=cached_ids==token;found=tl.sum(matches.to(tl.int32),axis=0)>0
    mass=tl.load(MASS+state)
    if (weight>0)&(context>=0)&(mass>0)&(mass<float('inf')):
        if found:
            q=tl.sum(tl.where(matches,tl.load(D_Q+(state//ROWS*CACHE+context)*K+k,k<K,other=0),0.),axis=0)
        else:
            h,r=tl.arange(0,BH),tl.arange(0,BR)
            x=tl.load(H+(state//ROWS*CACHE+context)*HIDDEN+h,h<HIDDEN,other=0).to(HEAD_DTYPE).to(tl.float32)
            row=tl.load(HEAD+token*HIDDEN+h,h<HIDDEN,other=0).to(HEAD_DTYPE).to(tl.float32)
            raw=tl.sum(x*row,axis=0)
            if HAS_BIAS:raw=raw+tl.load(BIAS+token).to(HEAD_DTYPE).to(tl.float32)
            # Dense SpecForge head returns the model dtype before FP32 softmax.
            raw=raw.to(HEAD_DTYPE).to(tl.float32)
            u=tl.load(U+(state//ROWS*CACHE+context)*R+r,r<R,other=0)
            adapter=tl.load(B+token*R+r,r<R,other=0)
            z=raw+tl.sum(adapter*u,axis=0)
            maximum=tl.load(NORM+(state//ROWS*CACHE+context)*2)
            total=tl.load(NORM+(state//ROWS*CACHE+context)*2+1)
            q=tl.div_rn(tl.exp(z-maximum),total)
    else:q=0.
    tl.store(OUT+state*K+j,q)


@triton.jit
def _union(TARGET,MAP,W,KIND,CONTEXTS,D_IDS,D_Q,T_IDS,T_P,T_Q,MASS,
            OUT_IDS,OUT_G,STATS,TS0,TS1,TS2,ROWS:tl.constexpr,CACHE:tl.constexpr,
            V:tl.constexpr,K:tl.constexpr,FIELDS:tl.constexpr,BU:tl.constexpr,GREEDY:tl.constexpr):
    state=tl.program_id(0).to(tl.int64);j=tl.arange(0,BU);first=j<K
    context=tl.maximum(tl.load(CONTEXTS+state),0);cache=state//ROWS*CACHE+context
    kk=j%K
    d_ids=tl.load(D_IDS+cache*K+kk,j<2*K,other=-1)
    token=tl.where(first,d_ids,tl.load(T_IDS+state*K+kk,j<2*K,other=-1))
    # Each top-k is unique. Only target additions require cross-set dedup.
    dk=tl.arange(0,triton.next_power_of_2(K))
    draft_ids=tl.load(D_IDS+cache*K+dk,dk<K,other=-2)
    overlap=tl.sum((token[:,None]==draft_ids[None,:]).to(tl.int32),axis=1)>0
    mass=tl.load(MASS+state);weight=tl.load(W+state)
    good=(mass>0)&(mass<float('inf'))
    valid=(j<2*K)&(first|~overlap)&(weight>0)&good
    target_id=tl.load(MAP+tl.maximum(token,0),j<2*K,other=0)
    if GREEDY:
        p=(target_id==tl.load(TARGET+state//ROWS*TS0+state%ROWS*TS1)).to(tl.float32)
    else:p=tl.load(TARGET+state//ROWS*TS0+state%ROWS*TS1+target_id*TS2,valid,other=0).to(tl.float32)
    p=tl.where(valid,tl.div_rn(p,tl.where(good,mass,1.)),0.)
    q=tl.where(first,tl.load(D_Q+cache*K+kk,j<2*K,other=0),tl.load(T_Q+state*K+kk,j<2*K,other=0))
    q=tl.where(valid,q,0.)
    p_tail=tl.maximum(1.-tl.sum(p,axis=0),0.);q_tail=tl.maximum(1.-tl.sum(q,axis=0),0.)
    empty_tail=tl.sum(valid.to(tl.int32),axis=0)==V
    p_tail=tl.where(empty_tail,0.,p_tail);q_tail=tl.where(empty_tail,0.,q_tail)
    # Exact categorical KL, with 0*log(0/q)=0. Infinite KL is not hidden.
    kl=tl.sum(tl.where(p>0,p*(tl.log(p)-tl.log(q)),0.),axis=0)
    kl=kl+tl.where(p_tail>0,p_tail*(tl.log(p_tail)-tl.log(q_tail)),0.)
    w=tl.where(good,weight,0.)
    tl.store(OUT_IDS+state*2*K+j,tl.where(valid,token,-1),j<2*K)
    tl.store(OUT_G+state*2*K+j,tl.where(valid,w*(q-p),0.),j<2*K)
    kind=tl.load(KIND+state);selected=w>0
    # Per-state summaries; one batched reduction feeds metrics and SGD denominator.
    tl.store(STATS+state*FIELDS+0,w)
    tl.store(STATS+state*FIELDS+1,selected.to(tl.float32))
    tl.store(STATS+state*FIELDS+2,(selected&(kind==1)).to(tl.float32))
    tl.store(STATS+state*FIELDS+3,(selected&(kind==2)).to(tl.float32))
    finite=(kl==kl)&(tl.abs(kl)<float('inf'))
    tl.store(STATS+state*FIELDS+4,tl.where(selected&finite,w*kl,0.))
    tl.store(STATS+state*FIELDS+5,tl.sum(valid.to(tl.float32),axis=0))
    tl.store(STATS+state*FIELDS+6,tl.where(selected,mass,0.))
    tl.store(STATS+state*FIELDS+7,tl.sum(tl.where(first,p,0.),axis=0))
    tl.store(STATS+state*FIELDS+8,((weight>0)&~good).to(tl.float32))
    tl.store(STATS+state*FIELDS+9,(selected&~finite).to(tl.float32))


@triton.jit
def _reduce(STATS,ROUND_WEIGHT,COUNTERS,N:tl.constexpr,FIELDS:tl.constexpr,BN:tl.constexpr):
    field=tl.program_id(0);n=tl.arange(0,BN)
    total=tl.sum(tl.load(STATS+n*FIELDS+field,n<N,other=0),axis=0)
    if field==0:tl.store(ROUND_WEIGHT,total)
    tl.atomic_add(COUNTERS+tl.where(field==9,12,field),total.to(tl.float64))


@triton.jit
def _update(IDS,G,U,CONTEXTS,ROUND_WEIGHT,B,BITS,ACTIVE,COUNT,
             ROWS:tl.constexpr,CACHE:tl.constexpr,K:tl.constexpr,R:tl.constexpr,
             LR:tl.constexpr,BU:tl.constexpr,BR:tl.constexpr):
    state=tl.program_id(0).to(tl.int64);j,r=tl.arange(0,BU),tl.arange(0,BR)
    token=tl.load(IDS+state*2*K+j,j<2*K,other=-1)
    g=tl.load(G+state*2*K+j,j<2*K,other=0)
    context=tl.maximum(tl.load(CONTEXTS+state),0)
    u=tl.load(U+(state//ROWS*CACHE+context)*R+r,r<R,other=0)
    denominator=tl.load(ROUND_WEIGHT)
    delta=-LR*tl.div_rn(g[:,None]*u[None,:],tl.where(denominator>0,denominator,1.))
    valid=(j<2*K)&(token>=0)&(denominator>0)
    tl.atomic_add(B+tl.maximum(token[:,None],0)*R+r[None,:],delta,valid[:,None]&(r[None,:]<R))
    changed=valid&(tl.sum((delta!=0).to(tl.int32),axis=1)>0)
    bit=1<<(token%32)
    old=tl.atomic_or(BITS+tl.maximum(token,0)//32,bit,changed)
    first=changed&((old&bit)==0)
    slot=tl.atomic_add(COUNT+tl.zeros((BU,),tl.int32),1,first)
    tl.store(ACTIVE+slot,token,first)


@triton.jit
def _round_end(B,BITS,ACTIVE,COUNT,WEIGHT,COUNTERS,R:tl.constexpr,
               LR:tl.constexpr,BS:tl.constexpr,BR:tl.constexpr):
    weight=tl.load(WEIGHT);count=tl.load(COUNT)
    s,r=tl.arange(0,BS),tl.arange(0,BR)
    kept=tl.full((),0,tl.int32)
    # Exact structural pruning, not gating: cancellation can make an adapter
    # row identically zero. Keep index/bitmap consistent, including B=0 identity.
    # ONE existing round-end CTA compacts in-place; no per-response state copy.
    for start in range(0,count,BS):
        token=tl.load(ACTIVE+start+s,start+s<count,other=0)
        w=tl.load(B+token[:,None]*R+r[None,:],(start+s[:,None]<count)&(r[None,:]<R),other=0)
        live=(tl.sum((w!=0).to(tl.int32),axis=1)>0)&(start+s<count)
        destination=kept+tl.cumsum(live.to(tl.int32),axis=0)-1
        tl.store(ACTIVE+tl.maximum(destination,0),token,live)
        tl.atomic_and(BITS+token//32,~(1<<(token%32)),(start+s<count)&~live)
        kept+=tl.sum(live.to(tl.int32),axis=0)
    tl.store(COUNT,kept)
    tl.atomic_add(COUNTERS+9,((weight>0)&(LR>0)).to(tl.float64))
    tl.atomic_add(COUNTERS+10,kept.to(tl.float64))
    tl.atomic_add(COUNTERS+11,1.)


def feedback(state,tree,path,target,greedy=False):
    b,q=tree.parents.shape;n=b*q;k=state.topk;rank=state.rank
    weights=state.selected_weights[:n].view(b,q);kind=state.selected_kind[:n].view(b,q)
    ticket=state.begin('opd_state_select_ms')
    select_states(tree,path,weights,kind,state.visited_weight,state.frontier_weight)
    state.end(ticket);ticket=state.begin('opd_teacher_extract_ms')
    probs=state.teacher_p[:n*k].view(n,k);ids=state.teacher_ids[:n*k].view(n,k);mass=state.teacher_mass[:n]
    teacher(target,state.mapping,weights,k,state.teacher_tiles,(probs,ids,mass),greedy)
    state.end(ticket);ticket=state.begin('opd_union_loss_ms')
    head=state.head.weight;bias=state.head.bias if state.head.bias is not None else head
    _selected_head[(k,n)](state.head_cache,head,bias,state.u_cache,state.B_fast,state.ids_cache,state.q_cache,
        ids,state.norm_cache,weights,mass,tree.feedback_contexts,state.teacher_q,
        triton.language.bfloat16 if state.logits_dtype==torch.bfloat16 else (triton.language.float16 if state.logits_dtype==torch.float16 else triton.language.float32),
        q,state.cache_contexts,head.shape[1],rank,k,state.head.bias is not None,triton.next_power_of_2(head.shape[1]),
        triton.next_power_of_2(rank),triton.next_power_of_2(k),num_warps=4,enable_fp_fusion=False)
    _union[(n,)](target,state.mapping,weights,kind,tree.feedback_contexts,state.ids_cache,state.q_cache,ids,probs,
        state.teacher_q,mass,state.union_ids,state.union_g,state.state_stats,
        target.stride(0),target.stride(1),0 if greedy else target.stride(2),q,state.cache_contexts,state.vocab,k,10,
        triton.next_power_of_2(2*k),greedy,num_warps=4,enable_fp_fusion=False)
    _reduce[(10,)](state.state_stats,state.round_weight,state.counters,n,10,triton.next_power_of_2(n),num_warps=4)
    state.end(ticket);ticket=state.begin('opd_update_ms')
    if state.fast_lr>0:
        _update[(n,)](state.union_ids,state.union_g,state.u_cache,tree.feedback_contexts,state.round_weight,
            state.B_fast,state.bitmap,state.active_ids,state.active_count,q,state.cache_contexts,k,rank,
            state.fast_lr,triton.next_power_of_2(2*k),triton.next_power_of_2(rank),num_warps=4,enable_fp_fusion=False)
    _round_end[(1,)](state.B_fast,state.bitmap,state.active_ids,
        state.active_count,state.round_weight,state.counters,rank,state.fast_lr,128,triton.next_power_of_2(rank),num_warps=4)
    state.end(ticket)
