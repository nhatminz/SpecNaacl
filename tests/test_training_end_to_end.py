import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]


def assert_nested_equal(a,b,*,projector_moments=False,path=()):
    if torch.is_tensor(a):
        if projector_moments and path==('opd_projector',):
            # Newly executed atomic updates can change A by one FP32 ULP.
            torch.testing.assert_close(a,b,rtol=2*torch.finfo(a.dtype).eps,atol=0)
        elif projector_moments and path[:2]==('state',0) and path[-1] in ('exp_avg','exp_avg_sq'):
            # GPU atomic accumulation order is not deterministic. Only A's
            # newly computed moments admit roundoff. Restoration itself is
            # checked bit for bit in the subprocess before the next rollout.
            # Cancellation makes elementwise relative error unsuitable near
            # zero. Bound atomic roundoff by eight FP32 eps at tensor scale.
            atol=8*torch.finfo(a.dtype).eps*max(a.abs().max().item(),b.abs().max().item())
            torch.testing.assert_close(a,b,rtol=0,atol=atol)
        else:assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:assert_nested_equal(a[k],b[k],projector_moments=projector_moments,path=path+(k,))
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b):assert_nested_equal(x,y)
    else:assert a==b


@pytest.mark.skipif(not torch.cuda.is_available(),reason='actual GRPO entrypoint requires CUDA')
@pytest.mark.parametrize('method',['fastgrpo','opd_reflex'])
def test_train_resume_restores_weights_optimizers_rng_metrics_and_cadence(tmp_path,method):
    script=ROOT/'tests/tiny_training_runner.py'
    def run(name,*args):
        with (tmp_path/(name+str(len(args))+'.log')).open('w') as log:
            result=subprocess.run([sys.executable,str(script),str(tmp_path),method,name,*args],
                stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'TQDM_DISABLE':'1'},timeout=120)
        assert result.returncode==0,(tmp_path/(name+str(len(args))+'.log')).read_text()[-8000:]
    run('full')
    run('resumed','--max_grpo_steps','1')
    events=json.loads((tmp_path/'resumed/test_events.json').read_text())
    assert events==['rollout','draft_backward','draft_step','target_step']
    run('resumed','--resume_checkpoint',str(tmp_path/'resumed/resume/latest.pt'))
    assert json.loads((tmp_path/'resumed/test_restore.json').read_text())['bitwise_restore']
    assert json.loads((tmp_path/'resumed/test_events.json').read_text())==events
    a=torch.load(tmp_path/'full/resume/latest.pt',map_location='cpu',weights_only=False)
    b=torch.load(tmp_path/'resumed/resume/latest.pt',map_location='cpu',weights_only=False)
    for key in ('target_lora','draft_model','optimizer_target','optimizer_draft',
                'step','draft_step','draft_accumulated_step','epoch','next_batch','used_items'):
        assert_nested_equal(a[key],b[key],projector_moments=method=="opd_reflex" and key in ("optimizer_draft","draft_model"))
    for key in ('torch','cuda','python'):
        assert_nested_equal(a['rank_states'][0]['rng'][key],b['rank_states'][0]['rng'][key])
    for name in ('_rollout_metrics_state',):
        ac=a['rank_states'][0]['batch_data'][name];bc=b['rank_states'][0]['batch_data'][name]
        for k in ac:
            if k!='generation':assert ac[k]==bc[k]
    for filename,column in [('timing.csv','step'),('rollout_timing.csv','global_iter')]:
        rows=list(csv.DictReader((tmp_path/'resumed/logs'/filename).open()))
        assert [int(row[column]) for row in rows]==[1,2]
    if method=='opd_reflex':assert 'opd_projector' in b['draft_model']
    else:assert 'opd_projector' not in b['draft_model']


@pytest.mark.skipif(not torch.cuda.is_available(),reason='actual pretraining entrypoint requires CUDA')
def test_pretrain_resume_weights_optimizer_scheduler_rng_and_paths(tmp_path):
    env={**os.environ,'TQDM_DISABLE':'1'}
    subprocess.run([sys.executable,str(ROOT/'tests/tiny_training_runner.py'),str(tmp_path)],check=True,env=env,stdout=subprocess.DEVNULL)
    def run(name,*extra):
        out=tmp_path/name
        command=[sys.executable,str(ROOT/'train_draft.py'),'--model_dir',str(tmp_path/'model'),
            '--dataset_dir',str(tmp_path/'sharegpt.json'),'--saved_model_dir',str(out/'checkpoints'),
            '--log_dir',str(out/'logs'),'--model_output_root',str(out),'--version_name','tiny',
            '--num_epochs','2','--batch_size','2','--num_workers','0','--max_length','128',
            '--save_interval','1',*extra]
        with (tmp_path/(name+'.log')).open('w') as log:
            result=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=120)
        assert result.returncode==0,(tmp_path/(name+'.log')).read_text()[-8000:]
        return out
    full=run('full')
    resumed=run('resumed','--max_steps','1')
    assert not (resumed/'checkpoints/pretrain_complete.json').exists()
    run('resumed','--resume','auto')
    a=torch.load(full/'latest_checkpoint/training_state.pt',weights_only=False)
    b=torch.load(resumed/'latest_checkpoint/training_state.pt',weights_only=False)
    for key in ('draft_model','optimizer','scheduler','step','accumulated','epoch','next_batch'):
        assert_nested_equal(a[key],b[key])
    for key in ('torch','cuda','python'):
        assert_nested_equal(a['ranks'][0]['rng'][key],b['ranks'][0]['rng'][key])
    assert b['step']==4 and b['epoch']==2
    assert (resumed/'latest_checkpoint/draft.pth').exists()
    assert (resumed/'latest_target_config.json').exists()
    assert (resumed/'checkpoints/pretrain_complete.json').exists()
    assert [json.loads(line)['step'] for line in (resumed/'logs/metrics.jsonl').read_text().splitlines()]==[1,2,3,4]
