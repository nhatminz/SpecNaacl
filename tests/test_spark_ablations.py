"""SPARK ablations: controlled traces, original-engine parity and real pipeline."""
import ast
from copy import deepcopy
import csv
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from helper.fastgrpo_model import FastGRPOModel
from helper.opd_ablation import (ablation_config, feedback_due, final_provenance,
    initialize_provenance, round_metrics, validate_resume_config)
from helper.opd_optimizer import draft_optimizer
from helper.opd_reflex import OPDReflex, initialize_projector, select_states_reference
from helper.tree_verification import PackedTree

ROOT = Path(__file__).resolve().parents[1]
DEVICES = ['cpu'] + (['cuda:0'] if torch.cuda.is_available() else [])


def config(**changes):
    values = dict(opd_projector_init='random_orthogonal', opd_projector_seed=42,
        opd_train_projector='1', opd_update_interval_rounds=1, opd_ablation_name='',
        opd_rank=8, opd_topk=16, opd_fast_lr=.01, opd_projector_lr=.0001,
        opd_visited_weight=1., opd_frontier_weight=1.)
    values.update(changes)
    return ablation_config(SimpleNamespace(**values))


@pytest.mark.parametrize('device', DEVICES)
def test_local_random_orthogonal_identical_starts_and_global_rng_unchanged(device):
    torch.manual_seed(923)
    before = torch.get_rng_state().clone()
    cuda_before = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    learned = initialize_projector(64, 8, seed=42, init='random_orthogonal').to(device)
    frozen = initialize_projector(64, 8, seed=42, init='random_orthogonal').to(device)
    assert torch.equal(learned, frozen)
    assert torch.equal(before, torch.get_rng_state())
    for x, y in zip(cuda_before, torch.cuda.get_rng_state_all() if cuda_before else []):
        assert torch.equal(x, y)
    torch.testing.assert_close(learned.T @ learned, torch.eye(8, device=device), atol=4e-7, rtol=1e-6)
    assert not torch.equal(learned.cpu(), initialize_projector(64, 8, seed=43, init='random_orthogonal'))


@pytest.mark.parametrize('interval', [0, 1, 5, 10, 15])
@pytest.mark.parametrize('length', [0, 1, 4, 5, 6, 11, 16, 31])
def test_round_schedule_resets_and_has_no_catchup(interval, length):
    expected = [] if interval == 0 else list(range(1, length + 1, interval))
    for _ in range(2):
        assert [r for r in range(1, length + 1) if feedback_due(r, interval)] == expected


def state(device, train, interval=1):
    from test_opd_reflex import state as fixture
    engine, model, mapping = fixture(device, v=37, lr=.02, topk=4, enabled=interval != 0)
    model.opd_projector = torch.nn.Parameter(initialize_projector(32, 8, init='random_orthogonal').to(device), requires_grad=train and interval != 0)
    engine.train_projector = train and interval != 0
    engine._layout = None
    engine.start(model, 3, mapping, 32, max_contexts=8, max_nodes=24, max_path=5, max_proposal_contexts=4)
    return engine, model, mapping


@pytest.mark.parametrize('device', DEVICES)
def test_fixed_verification_trace_learned_frozen_selection_teacher_and_b_parity(device):
    learned, lm, mapping = state(device, True)
    frozen, fm, _ = state(device, False)
    assert torch.equal(lm.opd_projector, fm.opd_projector)
    initial = lm.opd_projector.clone()
    generator = torch.Generator(device=device).manual_seed(54)
    hidden = torch.randn(3, 3, 32, generator=generator, device=device)
    teacher = torch.randn(3, 3, 37, generator=generator, device=device).softmax(-1)
    parents = torch.tensor([[-1, 0, 0]] * 3, device=device)
    tree = PackedTree(parents, parents.clone(), torch.tensor([[0, 1, 2]] * 3, device=device), 1)
    path = SimpleNamespace(packed_indices=torch.tensor([[0, 1, -1]] * 3, device=device))
    selected = select_states_reference(tree, path)
    for _ in range(3):
        metadata = []
        for engine, model in ((learned, lm), (frozen, fm)):
            engine.propose(model.lm_head(hidden), hidden, 3, mapping, root=True)
            small = engine.prepare_compact_teacher(tree, path, teacher)
            metadata.append(tuple(x.clone() for x in small))
            engine.feedback(tree, path, None, sampling_metadata=small)
        for x, y in zip(*metadata):
            assert torch.equal(x, y)
        torch.testing.assert_close(learned.B_fast, frozen.B_fast, rtol=2e-6, atol=2e-8)
        assert torch.equal(lm.opd_projector, initial)  # no per-round A update
        assert torch.equal(fm.opd_projector, initial)
        for x, y in zip(selected, select_states_reference(tree, path)):
            assert torch.equal(x, y)
    assert lm.opd_projector_grad_weight > 0
    assert lm.opd_projector_grad_sum.abs().sum() > 0
    assert not hasattr(fm, 'opd_projector_grad_sum')


@pytest.mark.parametrize('train,interval', [(True, 1), (False, 1), (True, 0)])
def test_optimizer_boundary_learned_changes_frozen_and_inactive_never_decay(train, interval):
    draft = torch.nn.Linear(32, 32)
    draft.register_parameter('opd_projector', torch.nn.Parameter(initialize_projector(32, 8, init='random_orthogonal'), requires_grad=train and interval != 0))
    draft.register_buffer('opd_projector_grad_sum', torch.ones_like(draft.opd_projector), persistent=False)
    draft.register_buffer('opd_projector_grad_weight', torch.ones(1), persistent=False)
    model = SimpleNamespace(draft_model=draft, opd_projector=draft.opd_projector,
        opd_projector_grad_sum=draft.opd_projector_grad_sum, opd_projector_grad_weight=draft.opd_projector_grad_weight)
    initialize_provenance(model)
    opt = draft_optimizer(draft, .0001, .001)
    active = train and interval != 0
    assert any(p is draft.opd_projector for g in opt.param_groups for p in g['params']) == active
    for _ in range(3):
        FastGRPOModel.apply_opd_projector_gradient(model)
        if not active:
            assert draft.opd_projector.grad is None
            # Optimizer exclusion also protects against accidental external grad.
            draft.opd_projector.grad = torch.ones_like(draft.opd_projector)
        draft(torch.ones(1, 32)).sum().backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    report = final_provenance(model)
    assert (report['opd_a_change_norm'] > 0) == active
    assert (report['opd_initial_a_sha256'] == report['opd_final_a_sha256']) == (not active)


@pytest.mark.parametrize('field,value', [('opd_train_projector','0'), ('opd_update_interval_rounds',5),
    ('opd_projector_seed',43), ('opd_projector_init','head_aligned'), ('opd_rank',4), ('opd_fast_lr',.02)])
def test_resume_rejects_different_ablation(field, value):
    with pytest.raises(ValueError, match='ablation configuration mismatch'):
        validate_resume_config(config(), config(**{field:value}))
    validate_resume_config(config(), config())
    with pytest.raises(ValueError, match='legacy'):
        validate_resume_config(None, config())
    validate_resume_config(None, config(opd_projector_init='head_aligned'))


@pytest.mark.parametrize('name,interval,train', [('learned_random',1,'1'),('frozen_random',1,'0'),
    ('interval5',5,'1'),('interval10',10,'1'),('interval15',15,'1'),('interval0',0,'1')])
def test_ablation_launcher_dry_run_and_independent_roots(name, interval, train, tmp_path):
    env = {k:v for k,v in os.environ.items() if k not in ('RUN_DIR','RUN_NAME','OPD_ABLATION_NAME')}
    env.update(DRY_RUN='true', OUTPUT_ROOT=str(tmp_path), PYTHON_BIN=sys.executable,
        DRAFT_CHECKPOINT='/fixed/pretrained-draft', MODEL='/fixed/Qwen2.5-3B-Instruct', RESUME='auto')
    out = subprocess.check_output(['bash', str(ROOT/'scripts/run_spark_ablation.sh'),name], env=env, text=True)
    tokens = shlex.split(next(line.split(':',1)[1] for line in out.splitlines() if line.startswith('Command')))
    from test_launchers_rewrite import flags
    opts = flags(tokens)
    assert opts['--opd_projector_init'] == 'random_orthogonal'
    assert opts['--opd_projector_seed'] == '42'
    assert opts['--opd_train_projector'] == train
    assert opts['--opd_update_interval_rounds'] == str(interval)
    assert opts['--max_target_optimizer_steps'] == '1000'
    assert opts['--train_option'] == 'simplelr_abel_level3to5'
    assert opts['--adapter_path'] == '/fixed/pretrained-draft'
    assert f'__train{train}__interval{interval}/' in opts['--log_file']
    assert '/ablations/' in opts['--log_file']
    assert opts['--resume_checkpoint'] == ''


def test_host_round_metrics_safe_zero_and_actual_updates_not_feedback_attempts():
    assert round_metrics({})['opd_actual_update_frequency'] == 0
    result = round_metrics(dict(verification_batches=11, opd_feedback_rounds=3, opd_skipped_rounds=8, opd_updates=2))
    assert result['opd_actual_update_frequency'] == 2/11
    assert result['opd_feedback_frequency'] == 3/11


@pytest.mark.skipif(not torch.cuda.is_available(), reason='real CUDA OPD rollout')
@pytest.mark.parametrize('stream',[False,True])
def test_interval_one_bitwise_original_engine_output_rng_and_projector_gradient(stream):
    from test_fastgrpo_rewrite import tiny, run
    from helper.opd_generate import speculative_generate
    spec = importlib.util.spec_from_file_location('opd_before_ablation',ROOT/'tests/references/opd_generate_before_ablation.py')
    old = importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    model = tiny();model.enable_opd(8)
    other = deepcopy(model)
    a = run(old.speculative_generate, model, opd_update_stream=stream, opd_train_projector=True)
    b = run(speculative_generate, other, opd_update_stream=stream, opd_train_projector=True, opd_update_interval_rounds=1)
    for field in ('generated_token_ids','total_acc_length','total_decoded_token_num','response_verification_rounds'):
        assert a[0][field] == b[0][field]
    assert torch.equal(a[1], b[1])
    assert a[2] == b[2]
    for x,y in zip(a[3],b[3]):
        assert torch.equal(x,y)
    torch.testing.assert_close(model.opd_projector_grad_sum, other.opd_projector_grad_sum, atol=2e-7, rtol=2e-5)
    assert torch.equal(model.opd_projector_grad_weight, other.opd_projector_grad_weight)
    from helper.opd_reflex import OPD_COUNTER_NAMES
    for field in OPD_COUNTER_NAMES:
        assert a[0][field] == pytest.approx(b[0][field], abs=2e-7, rel=2e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='real CUDA schedule and async dependencies')
@pytest.mark.parametrize('interval',[0,1,5,10,15])
@pytest.mark.parametrize('stream',[False,True])
@pytest.mark.parametrize('eos',[False,True])
def test_actual_generation_update_rounds_teacher_skips_reset_and_eos(interval, stream, eos, monkeypatch):
    import helper.opd_generate as generation
    from test_fastgrpo_rewrite import tiny
    model = tiny();model.enable_opd(8, 'random_orthogonal', 42)
    rounds = [0];updates=[];teachers=[]
    sample = generation.sample_target_with_metadata
    feedback = OPDReflex.feedback
    prepare = OPDReflex.prepare_compact_teacher
    def sampler(logits, **kw):
        rounds[0] += 1
        if eos:
            logits = torch.full_like(logits, -50.);logits[...,96] = 50.
        return sample(logits, **kw)
    def feedback_spy(self,*a,**kw):
        updates.append(rounds[0]-1)
        return feedback(self,*a,**kw)
    def teacher_spy(self,*a,**kw):
        teachers.append(rounds[0]-1)
        return prepare(self,*a,**kw)
    monkeypatch.setattr(generation,'sample_target_with_metadata',sampler)
    monkeypatch.setattr(OPDReflex,'feedback',feedback_spy)
    monkeypatch.setattr(OPDReflex,'prepare_compact_teacher',teacher_spy)
    for _ in range(2):
        rounds[0]=0;updates.clear();teachers.clear()
        with torch.inference_mode():
            out=generation.speculative_generate(model,torch.tensor([[3,5,8]]),torch.ones(1,3,dtype=torch.long),
                SimpleNamespace(eos_token_id=96),do_sample=True,temperature=.8,top_p=.95,
                max_length=40,verification_capacity=14,max_verification_num=7,max_draft_k=2,
                max_draft_token_length=3,min_draft_token_length=3,opd_update_stream=stream,
                opd_train_projector=True,opd_update_interval_rounds=interval)
        n=out['verification_batches']
        expected=[] if interval==0 else list(range(1,n+1,interval))
        assert updates==teachers==expected
        assert out['opd_feedback_rounds']==len(expected)
        assert out['opd_skipped_rounds']==n-len(expected)
        assert out['opd_updates']==len(expected)
        if eos:assert n==1
        engine=next(iter(model._opd_runtime_cache.values()))
        assert not engine._ever_updated  # clear() at rollout end
        if interval==0:
            assert engine.B_fast is None  # disabled adapter is identically zero
            assert not model.opd_projector_grad_sum.any()
            assert not model.opd_projector_grad_weight.any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='actual training CLI + checkpoint')
@pytest.mark.parametrize('train,interval',[('1',1),('0',1),('1',5),('1',0)])
def test_training_pipeline_logs_projector_and_resume_guard(tmp_path, train, interval):
    extra=['--opd_projector_init','random_orthogonal','--opd_projector_seed','42',
           '--opd_train_projector',train,'--opd_update_interval_rounds',str(interval),
           '--max_target_optimizer_steps','1']
    command=[sys.executable,str(ROOT/'tests/tiny_training_runner.py'),str(tmp_path),'opd_reflex','run',*extra]
    env={**os.environ,'TQDM_DISABLE':'1'}
    result=subprocess.run(command,env=env,text=True,capture_output=True,timeout=120)
    assert result.returncode==0,result.stdout[-3000:]+result.stderr[-4000:]
    summary=json.loads((tmp_path/'run/summary.json').read_text())
    assert summary['target_optimizer_steps']==1 and summary['draft_step']>0
    changed=train=='1' and interval>0
    assert (summary['opd_a_change_norm']>0)==changed
    assert (summary['opd_initial_a_sha256']!=summary['opd_final_a_sha256'])==changed
    assert summary['total_verification_rounds']==summary['opd_feedback_rounds']+summary['opd_skipped_rounds']
    rows=list(csv.DictReader((tmp_path/'run/logs/rollout_timing.csv').open()))
    assert rows and rows[0]['opd_projector_init']=='random_orthogonal'
    steps=list(csv.DictReader((tmp_path/'run/logs/timing.csv').open()))
    assert steps and steps[0]['opd_projector_init']=='random_orthogonal'
    assert steps[-1]['opd_final_a_sha256']==summary['opd_final_a_sha256']
    assert float(steps[-1]['opd_a_change_norm'])==summary['opd_a_change_norm']
    assert float(steps[0]['cumulative_opd_feedback_rounds'])==summary['opd_feedback_rounds']
    checkpoint=tmp_path/'run/resume/latest.pt'
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert payload['opd_ablation_config']['opd_update_interval_rounds']==interval
    assert payload['opd_initial_a_sha256']==summary['opd_initial_a_sha256']
    # Cross-ablation resume fails before loading optimizer state or new rollout.
    wrong=command+['--resume_checkpoint',str(checkpoint),'--opd_update_interval_rounds','10']
    result=subprocess.run(wrong,env=env,text=True,capture_output=True,timeout=120)
    assert result.returncode!=0
    assert 'ablation configuration mismatch' in result.stderr
    right=command+['--resume_checkpoint',str(checkpoint),'--max_target_optimizer_steps','2']
    result=subprocess.run(right,env=env,text=True,capture_output=True,timeout=120)
    assert result.returncode==0,result.stdout[-3000:]+result.stderr[-4000:]
    resumed=json.loads((tmp_path/'run/summary.json').read_text())
    assert resumed['target_optimizer_steps']==2
    assert resumed['opd_initial_a_sha256']==summary['opd_initial_a_sha256']
    assert (resumed['opd_a_change_norm']>0)==changed
    assert resumed['opd_feedback_rounds']>=summary['opd_feedback_rounds']
