"""Five-study invariants: grouped oracle, row ownership, carry and real CUDA."""
from copy import deepcopy
from types import SimpleNamespace
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from helper.opd_reflex import OPDReflex, capture_persistent_b, restore_persistent_b, select_states_reference
from helper.tree_verification import PackedTree

ROOT=Path(__file__).resolve().parents[1]
DEVICES=['cpu']+(['cuda:0'] if torch.cuda.is_available() else [])


def fixture(device,scope='rollout_batch',reset='rollout',batch=4,groups=2,repeats=2):
    from test_opd_reflex import state
    _,model,mapping=state(device,v=37,topk=4)
    engine=OPDReflex(topk=4,fast_lr=.02,train_projector=True,
        b_sharing_scope=scope,b_reset_mode=reset)
    start(engine,model,mapping,batch,groups,repeats)
    return engine,model,mapping


def start(engine,model,mapping,batch=4,groups=2,repeats=2):
    engine.start(model,batch,mapping,32,max_contexts=8,max_nodes=32,max_path=5,
        max_proposal_contexts=4,prompt_groups=groups,responses_per_prompt=repeats)


def trace(device,batch):
    parents=torch.tensor([[-1,0,0]]*batch,device=device)
    tree=PackedTree(parents,parents.clone(),torch.tensor([[0,1,2]]*batch,device=device),1)
    path=SimpleNamespace(packed_indices=torch.tensor([[0,1,-1]]*batch,device=device))
    return tree,path


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('mode',['sparse','fused','gemm'])
def test_grouped_matches_independent_feedback_and_correction(device,mode,monkeypatch):
    monkeypatch.setenv('OPD_PROPOSAL_MODE','sparse' if mode=='sparse' else 'dense')
    monkeypatch.setenv('OPD_DENSE_IMPLEMENTATION','fused' if mode=='sparse' else mode)
    grouped,model,mapping=fixture(device,'per_prompt_group')
    separate=[]
    for _ in range(2):
        e=OPDReflex(topk=4,fast_lr=.02,train_projector=True)
        m=deepcopy(model);start(e,m,mapping,2,1,2);separate.append((e,m))
    gen=torch.Generator(device=device).manual_seed(79)
    hidden=torch.randn(4,3,32,generator=gen,device=device)
    teacher=torch.randn(4,3,37,generator=gen,device=device).softmax(-1)
    tree,path=trace(device,4);small_tree,small_path=trace(device,2)
    for round_index in range(3):
        q,ids,_=grouped.propose(model.lm_head(hidden),hidden,4,mapping,root=True)
        q=q.clone();ids=ids.clone()
        old=grouped.B_fast.clone()
        if round_index==1:teacher[2:].zero_()  # group with no valid teacher must stay unchanged
        grouped.feedback(tree,path,teacher)
        if round_index==1:assert torch.equal(old[1],grouped.B_fast[1])
        for g,(engine,m) in enumerate(separate):
            rows=slice(2*g,2*g+2)
            sq,si,_=engine.propose(m.lm_head(hidden[rows]),hidden[rows],4,mapping,root=True)
            assert torch.equal(ids[rows],si)
            torch.testing.assert_close(q[rows],sq,rtol=3e-6,atol=1e-7)
            engine.feedback(small_tree,small_path,teacher[rows])
            torch.testing.assert_close(grouped.B_fast[g],engine.B_fast,rtol=3e-5,atol=2e-8)
        torch.testing.assert_close(model.opd_projector_grad_sum,
            sum(m.opd_projector_grad_sum for _,m in separate),rtol=5e-5,atol=2e-7)
        torch.testing.assert_close(model.opd_projector_grad_weight,
            sum(m.opd_projector_grad_weight for _,m in separate))


@pytest.mark.parametrize('device',DEVICES)
def test_group_identity_after_swap_remove_and_canonical_feedback(device):
    grouped,model,mapping=fixture(device,'per_prompt_group',batch=6,groups=3,repeats=2)
    # Swap-remove responses 0 and 3: physical owners [5,1,2,4], canonical [1,2,4,5].
    owners=[5,1,2,4];keep=torch.tensor(owners,device=device)
    grouped.compact_group_rows(keep,owners)
    assert grouped.row_groups.tolist()==[2,0,1,2]
    assert grouped.group_first.tolist()==[1,2,0]
    grouped.feedback_row_map=torch.tensor([1,2,3,0],device=device)
    gen=torch.Generator(device=device).manual_seed(55)
    hidden=torch.randn(4,3,32,generator=gen,device=device)
    teacher=torch.zeros(4,3,37,device=device)
    teacher[0]=torch.randn(3,37,generator=gen,device=device).softmax(-1) # canonical response 1 -> prompt 0
    tree,path=trace(device,4)
    grouped.propose(model.lm_head(hidden),hidden,4,mapping,root=True)
    grouped.feedback(tree,path,teacher)
    assert grouped.B_fast[0].abs().sum()>0
    assert not grouped.B_fast[1:].any()
    grouped.propose(model.lm_head(hidden),hidden,4,mapping,root=True)
    dense=model.lm_head(hidden).float()+torch.bmm(grouped.u_cache[:4,:3],grouped.B_fast[grouped.row_groups].transpose(1,2))
    expected=torch.argsort(dense,descending=True,stable=True)[...,:4]
    assert torch.equal(grouped.ids_cache[:4,:3],expected)


@pytest.mark.parametrize('device',DEVICES)
def test_carry_boundary_restore_cache_resize_and_projector_change(device):
    engine,model,mapping=fixture(device,reset='carry')
    hidden=torch.randn(4,3,32,device=device);tree,path=trace(device,4)
    engine.propose(model.lm_head(hidden),hidden,4,mapping,root=True)
    engine.feedback(tree,path,torch.randn(4,3,37,device=device).softmax(-1))
    before=engine.B_fast.clone();engine.finish();engine.clear()
    state=capture_persistent_b(model)
    assert set(state)=={'B_fast','bitmap','active_ids','active_count','ever_updated','host_active_count'}
    assert state['ever_updated'] and state['active_count']>0
    # A/target boundary changes do not reset carry B. Change batch/layout too.
    model.opd_projector.add_(.001)
    start(engine,model,mapping,2,1,2)
    assert torch.equal(engine.B_fast,before)
    replacement=deepcopy(model);del replacement._opd_persistent_b
    restore_persistent_b(replacement,deepcopy(state))
    other=OPDReflex(topk=4,fast_lr=.02,train_projector=True,b_reset_mode='carry')
    start(other,replacement,mapping,2,1,2)
    for name in ('B_fast','bitmap','active_ids','active_count'):
        assert torch.equal(getattr(engine,name),getattr(other,name))
    for m,e in ((model,engine),(replacement,other)):
        result=e.propose(m.lm_head(hidden[:2]),hidden[:2],4,mapping,root=True)
        if e is engine:expected=[x.clone() for x in result]
        else:
            for x,y in zip(expected,result):assert torch.equal(x,y)
    fresh=deepcopy(model);del fresh._opd_persistent_b
    cold=OPDReflex(topk=4,b_reset_mode='carry');start(cold,fresh,mapping)
    assert not cold.B_fast.any() and not cold._ever_updated
    normal,_,_=fixture(device)
    normal.B_fast.fill_(1);normal.clear();assert not normal.B_fast.any()


@pytest.mark.parametrize('device',DEVICES)
def test_accepted_only_reuses_visited_selection_and_excludes_frontier(device):
    engine,model,mapping=fixture(device)
    engine.frontier_weight=0
    tree,path=trace(device,4)
    w,kind=select_states_reference(tree,path,1.,0.)
    assert kind.tolist()==[[1,1,0]]*4
    hidden=torch.randn(4,3,32,device=device)
    teacher=torch.zeros(4,3,37,device=device)
    teacher[:,2]=1/37 # only frontier has teacher => no update
    engine.propose(model.lm_head(hidden),hidden,4,mapping,root=True)
    engine.feedback(tree,path,teacher)
    assert not engine.B_fast.any()
    result=engine.finish()
    assert result['opd_frontier_states']==result['opd_updates']==0


@pytest.mark.skipif(not torch.cuda.is_available(),reason='real transformer + Triton')
@pytest.mark.parametrize('scope,reset',[('per_prompt_group','rollout'),('rollout_batch','carry')])
@pytest.mark.parametrize('stream',[False,True])
def test_real_generation_group_compaction_carry_and_no_extra_target_forward(scope,reset,stream,monkeypatch):
    from test_fastgrpo_rewrite import tiny,run
    import helper.opd_generate as generation
    model=tiny();model.enable_opd(8,'random_orthogonal',42)
    sample=generation.sample_target_with_metadata
    calls=[0];seen=[];feedback=OPDReflex.feedback
    def sampler(logits,**kwargs):
        calls[0]+=1
        if calls[0]==2:
            logits=logits.clone();logits[0].fill_(-50);logits[0,...,96]=50
        return sample(logits,**kwargs)
    def spy(self,*args,**kwargs):
        if self.grouped:seen.append(self.row_groups.cpu().tolist()) # test-only observer
        return feedback(self,*args,**kwargs)
    monkeypatch.setattr(generation,'sample_target_with_metadata',sampler)
    monkeypatch.setattr(OPDReflex,'feedback',spy)
    kwargs=dict(opd_update_stream=stream,opd_train_projector=True,opd_b_sharing_scope=scope,opd_b_reset_mode=reset)
    result=run(generation.speculative_generate,model,**kwargs)
    assert result[2]['target']==result[0]['verification_batches']+1
    assert result[0]['opd_updates']>0
    if scope=='per_prompt_group':
        assert seen[0]==[0,0,1,1]
        assert seen[1]==[1,0,1] # original owners after response 0 EOS
    else:
        state=capture_persistent_b(model)
        assert state['B_fast'].abs().sum()>0
        pointer=model._opd_persistent_b['B_fast'].data_ptr()
        calls[0]=0
        run(generation.speculative_generate,model,**kwargs)
        assert model._opd_persistent_b['B_fast'].data_ptr()==pointer


@pytest.mark.skipif(not torch.cuda.is_available(),reason='training CLI checkpoint smoke')
@pytest.mark.parametrize('name,extra',[
    ('interval5',['--opd_update_interval_rounds','5']),
    ('interval10',['--opd_update_interval_rounds','10']),
    ('per_prompt_group',['--opd_b_sharing_scope','per_prompt_group']),
    ('carry_b',['--opd_b_reset_mode','carry']),
    ('accepted_only',['--opd_frontier_weight','0.'])])
def test_five_training_logs_and_resume(tmp_path,name,extra):
    common=['--opd_projector_init','random_orthogonal','--opd_projector_seed','42',
        '--opd_ablation_name',name,*extra]
    command=[sys.executable,str(ROOT/'tests/tiny_training_runner.py'),str(tmp_path),'opd_reflex','run',*common]
    env={**os.environ,'TQDM_DISABLE':'1'}
    def launch(flags):
        p=subprocess.run(command+flags,env=env,text=True,capture_output=True,timeout=120)
        assert p.returncode==0,p.stdout[-3000:]+p.stderr[-5000:]
    launch(['--max_target_optimizer_steps','1'])
    path=tmp_path/'run/resume/latest.pt'
    payload=torch.load(path,map_location='cpu',weights_only=False)
    summary=json.loads((tmp_path/'run/summary.json').read_text())
    cfg=payload['opd_ablation_config']
    assert cfg['opd_ablation_name']==name
    for key in ('opd_ablation_name','opd_b_sharing_scope','opd_b_reset_mode','opd_feedback_selection'):
        assert summary[key]==cfg[key]
        for filename in ('timing.csv','rollout_timing.csv'):
            rows=list(csv.DictReader((tmp_path/'run/logs'/filename).open()))
            assert rows and all(row[key]==str(cfg[key]) for row in rows)
    if name=='carry_b':
        fast=payload['rank_states'][0]['opd_fast_state']
        assert fast['B_fast'].abs().sum()>0 and fast['ever_updated']
        assert fast['active_count']>0
    launch(['--max_target_optimizer_steps','2','--resume_checkpoint',str(path)])
    assert json.loads((tmp_path/'run/test_restore.json').read_text())['bitwise_restore']
    summary=json.loads((tmp_path/'run/summary.json').read_text())
    assert summary['target_optimizer_steps']==2
    assert summary['opd_initial_a_sha256']==payload['opd_initial_a_sha256']
    if name=='accepted_only':assert summary['cumulative_opd_frontier_states']==0
