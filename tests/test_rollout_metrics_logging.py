"""Host-only iteration times/throughputs and append/resume schema correctness."""
import ast
import csv
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from helper import rollout_metrics
from helper.rollout_metrics import (FIELDS, ITER_TIME_FIELDS, DRAFT_UPDATE_FIELDS,
                                    RolloutMetricsWriter)


def rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


@pytest.mark.parametrize('method',['fastgrpo','opd_reflex'])
def test_iteration_clock_rates_host_phase_times_and_available_diagnostics(tmp_path,monkeypatch,method):
    stamps=iter([100.,104.])
    monkeypatch.setattr(rollout_metrics.time,'perf_counter',lambda:next(stamps))
    path=tmp_path/'rollout_timing.csv'
    writer=RolloutMetricsWriter(path,method)
    writer.begin(1,0,2,0)
    writer.finish(dict(total_acc_length=9,total_decoded_token_num=3,
        total_time_cost=2.,response_generated_tokens=[4,6],
        total_accepted_draft_tokens=6,total_proposed_draft_tokens=8,
        iter_draft_train_time_s=.5,iter_target_train_time_s=1.5,
        iter_draft_sparse_kl=.125,iter_draft_sparse_tv=.25,iter_draft_update_committed=True),
        grpo_step=1,used_items=2,wall_time_s=9999.,draft_updates_cumulative=7)
    writer.close()
    row=rows(path)[0]
    assert float(row['iter_wall_time_s'])==4.
    assert float(row['iter_generation_tokens_per_s'])==5.
    assert float(row['iter_end_to_end_tokens_per_s'])==2.5
    assert float(row['iter_draft_train_time_s'])==.5
    assert float(row['iter_target_train_time_s'])==1.5
    assert float(row['iter_draft_sparse_kl'])==.125
    assert float(row['iter_draft_sparse_tv'])==.25
    assert row['iter_draft_update_committed']=='True'
    assert row['iter_draft_updates_cumulative']=='7'
    # Existing counters/definitions remain intact; startup time is not used
    # as the denominator of per-iteration end-to-end throughput.
    assert float(row['cumulative_wall_time_s'])==9999.
    assert float(row['iter_aal'])==3.
    assert float(row['iter_acceptance_rate'])==.75
    assert row['method']==method


@pytest.mark.parametrize('method',['fastgrpo','opd_reflex'])
def test_zero_denominators_are_safe_and_skip_without_rollout_logs_once(tmp_path,monkeypatch,method):
    stamps=iter([10.,10.,20.,23.])
    monkeypatch.setattr(rollout_metrics.time,'perf_counter',lambda:next(stamps))
    path=tmp_path/'rollout_timing.csv'
    writer=RolloutMetricsWriter(path,method)
    writer.begin(1,0,2,0)
    writer.finish(dict(response_generated_tokens=[5],total_time_cost=0.),
        grpo_step=0,used_items=0,wall_time_s=100.)
    writer.begin(1,1,2,0)
    writer.finish(None,grpo_step=0,used_items=0,wall_time_s=103.,draft_updates_cumulative=3)
    writer.finish(None,grpo_step=0,used_items=0,wall_time_s=103.)
    writer.close()
    first,skipped=rows(path)
    assert float(first['iter_generation_tokens_per_s'])==0.
    assert float(first['iter_end_to_end_tokens_per_s'])==0.
    assert float(skipped['iter_wall_time_s'])==3.
    for name in ('iter_generation_tokens_per_s','iter_end_to_end_tokens_per_s',
                 'iter_draft_train_time_s','iter_target_train_time_s','iter_rollout_tokens'):
        assert float(skipped[name])==0.
    assert skipped['iter_draft_update_committed']=='False'
    assert skipped['iter_draft_updates_cumulative']=='3'
    assert skipped['iter_draft_sparse_kl']==skipped['iter_draft_sparse_tv']==''


def test_tensor_diagnostics_are_omitted_without_reads_and_buffering_is_unchanged(tmp_path,monkeypatch):
    path=tmp_path/'rollout_timing.csv'
    writer=RolloutMetricsWriter(path,'opd_reflex',flush_interval=10)
    tensor=torch.tensor(.3)
    def forbidden(*a,**k):pytest.fail('logging attempted a tensor read, GPU work, extra open or flush')
    with monkeypatch.context() as patch:
        for name in ('item','cpu','tolist','__float__','__bool__','sum'):
            patch.setattr(torch.Tensor,name,forbidden)
        for name in ('synchronize','Event'):
            patch.setattr(torch.cuda,name,forbidden)
        patch.setattr(Path,'open',forbidden)
        patch.setattr(writer,'flush',forbidden)
        writer.begin(1,0,2,0)
        writer.finish(dict(response_generated_tokens=[2,3],total_time_cost=1.,
            iter_draft_train_time_s=tensor,iter_target_train_time_s=tensor,
            iter_draft_sparse_kl=tensor,iter_draft_sparse_tv=tensor,
            iter_draft_update_committed=tensor),
            grpo_step=0,used_items=0,wall_time_s=2.,draft_updates_cumulative=tensor)
        assert writer.unflushed==1
    writer.close()
    row=rows(path)[0]
    for name in ('iter_draft_train_time_s','iter_target_train_time_s',*DRAFT_UPDATE_FIELDS):
        assert row[name]==''


@pytest.mark.parametrize('method',['fastgrpo','opd_reflex'])
def test_resume_migrates_old_schema_rewinds_and_keeps_existing_values(tmp_path,monkeypatch,method):
    path=tmp_path/'rollout_timing.csv'
    old_fields=FIELDS[:-len(ITER_TIME_FIELDS)-len(DRAFT_UPDATE_FIELDS)]
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=old_fields)
        writer.writeheader()
        writer.writerow(dict(global_iter=1,method=method,iter_aal=3,
                             cumulative_wall_time_s=11,iter_rollout_tokens=8))
        writer.writerow(dict(global_iter=2,method=method,iter_aal=4))
    state=dict(global_iter=1,accepted=9,rounds=3,tokens=8,generation=2.,
               accepted_draft=6,proposed=8)
    stamps=iter([200.,202.])
    monkeypatch.setattr(rollout_metrics.time,'perf_counter',lambda:next(stamps))
    writer=RolloutMetricsWriter(path,method,state=deepcopy(state))
    writer.begin(1,1,2,0)
    writer.finish(None,grpo_step=1,used_items=0,wall_time_s=13.,draft_updates_cumulative=8)
    writer.close()
    data=rows(path)
    assert list(data[0])==list(FIELDS)
    assert [row['global_iter'] for row in data]==['1','2']
    assert data[0]['iter_aal']=='3' and data[0]['iter_rollout_tokens']=='8'
    assert data[0]['cumulative_wall_time_s']=='11'
    assert all(data[0][name]=='' for name in (*ITER_TIME_FIELDS,*DRAFT_UPDATE_FIELDS))
    assert data[1]['iter_wall_time_s']=='2.0'
    assert data[1]['iter_draft_updates_cumulative']=='8'
    assert float(data[1]['cumulative_aal'])==3.
    assert path.with_suffix('.pre_resume.csv').exists()


def test_writer_has_no_gpu_calls_and_training_reuses_host_phase_values():
    root=Path(__file__).resolve().parents[1]
    tree=ast.parse((root/'helper/rollout_metrics.py').read_text())
    banned={'synchronize','item','cpu','tolist','Event','elapsed_time','query','nonzero'}
    calls={n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)}
    assert not (calls&banned)
    training=ast.parse((root/'grpo_speculative.py').read_text())
    assignments={ast.unparse(n.targets[0]):n.value for n in ast.walk(training)
        if isinstance(n,ast.Assign) and len(n.targets)==1}
    draft=assignments["iter_outputs['iter_draft_train_time_s']"]
    target=assignments["iter_outputs['iter_target_train_time_s']"]
    assert ast.unparse(draft)=='phase_timings.pending[-1][1]'
    assert 'phase_timings.pending[-1][1]' in ast.unparse(target)
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in banned
                   for expr in (draft,target) for n in ast.walk(expr))
