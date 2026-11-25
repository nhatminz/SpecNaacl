import csv
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from helper.fastgrpo_generate import get_adaptive_hyperparameters
from helper.generation_telemetry import GenerationTelemetry, cap_path, storage_bytes
from helper.rebuttal_metrics import length_metrics, reward_metrics
from helper.specualtive_generate import speculative_generate
from helper.tree_verification import VerifiedPath
from scripts.summarize_rebuttal import stat, comparison


ROOT = Path(__file__).resolve().parents[1]
TRAIN_LAUNCHERS = sorted((ROOT / 'scripts/rebuttal').glob('train_qwen*.sh'))


def launcher_command(script, **overrides):
    env = dict(os.environ, DRY_RUN='true', DATASET='gsm8k', CUDA_VISIBLE_DEVICES='3',
               NPROC_PER_NODE='1', TRAIN_SUBSET_SEED='42', RESUME='', RUN_NAME='', RUN_DIR='',
               **overrides)
    out = subprocess.check_output(['bash',str(script)], text=True, cwd=ROOT, env=env)
    return shlex.split(next(line for line in out.splitlines() if line.startswith('Command')).split(':',1)[1])


@pytest.mark.parametrize('script', TRAIN_LAUNCHERS, ids=lambda script: script.stem)
def test_seed_wrappers_separate_training_seed_from_data_seed(script,tmp_path):
    args = launcher_command(script, OUTPUT_ROOT=str(tmp_path))
    value = lambda flag: args[args.index(flag)+1]
    seed = '123' if '123' in script.stem else '456'
    assert value('--seed') == seed
    assert value('--train_subset_seed') == value('--prompt_order_seed') == '42'
    assert value('--method') == ('opd_reflex' if 'opd_reflex' in script.stem else 'fastgrpo')
    assert value('--train_option') == 'gsm8k'
    assert '/nlp/users/minhpn19/data/' in value('--dataset_path')
    assert value('--batch_size') == value('--repeated_generate_nums') == '8'
    assert '--nproc_per_node=1' in args and not value('--resume_checkpoint')
    assert str(tmp_path/'pretrain/qwen25_7b/latest_checkpoint') == value('--adapter_path')
    assert str(tmp_path/'rebuttal/multiseed') in value('--summary_file')
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('script', ['train_qwen25_3b.sh','train_qwen25_3b_fastgrpo.sh'])
@pytest.mark.parametrize('experiment,extra', [
    ('concurrency', {'RESPONSES_PER_PROMPT':'16', 'GENERATION_PROFILE':'detailed'}),
    ('long_context', {'MAX_RESPONSE_TOKENS':'4096'}),
])
def test_original_launchers_accept_response_overrides(script,experiment,extra,tmp_path):
    args = launcher_command(ROOT/script, OUTPUT_ROOT=str(tmp_path), REBUTTAL_EXPERIMENT=experiment, **extra)
    value = lambda flag: args[args.index(flag)+1]
    assert value('--rebuttal_experiment') == experiment
    assert value('--batch_size') == '8'
    assert value('--repeated_generate_nums') == extra.get('RESPONSES_PER_PROMPT','8')
    assert value('--generation_profile') == extra.get('GENERATION_PROFILE','clean')
    assert value('--max_length') == value('--max_prompt_length') == '2048'
    if experiment == 'long_context':
        assert value('--max_response_tokens') == '4096'
    else:
        assert '--max_response_tokens' not in args


def test_concurrency_keeps_verification_capacity_and_changes_budget():
    assert get_adaptive_hyperparameters(64,512,5,8,160,3,.75) == (3,7,7)
    assert get_adaptive_hyperparameters(128,512,5,8,160,3,.75) == (3,3,3)


def test_unconditional_rewards_include_zero_variance_and_format():
    groups = [([(1.,1.,0.),(1.2,1.,1.)],True), ([(.2,0.,1.),(.2,0.,1.)],False),
              ([(1.,1.,0.),(1.,1.,0.)],False)]
    stats = reward_metrics(groups)
    assert stats['reward_mean'] == pytest.approx(4.6/6)
    assert stats['eligible_reward_mean'] == pytest.approx(1.1)
    assert stats['correctness_mean'] == pytest.approx(4/6)
    assert stats['format_reward_mean'] == .5
    assert stats['all_correct_prompt_groups'] == 2
    assert stats['all_incorrect_prompt_groups'] == 1
    assert stats['zero_variance_prompt_groups'] == 2


def test_caps_are_per_response_and_keep_only_the_committed_path():
    field = torch.tensor([[9,12,13],[8,3,6],[4,5,-1]])
    path = VerifiedPath(field.clone(),field.clone(),field.clone(),torch.tensor([3,3,2]))
    assert cap_path(path, [0, 1, 3]).lengths.tolist() == [0,1,2]
    assert path.tokens.tolist() == [[-1,-1,-1],[8,-1,-1],[4,5,-1]]
    assert torch.equal(path.tokens,path.packed_indices)
    stats = length_metrics([1,3000,4096], [[9],[8],[7]], 9, 2048, 4096, [500], 3)
    assert stats['generated_tokens'] == 7097
    assert stats['fraction_over_2048'] == 2/3
    assert stats['fraction_reaching_4096'] == 1/3
    assert stats['eos_rate'] == stats['length_cap_rate'] == 1/3


def test_profile_aggregate_does_not_depend_on_round_sampling():
    telemetry = GenerationTelemetry('detailed','cpu',2,2)
    trial = torch.zeros(3,4)
    assert storage_bytes([trial,trial.view(-1)[:4]]) == 48
    for i in range(1,5):
        ticket = telemetry.begin('draft_proposal')
        telemetry.end(ticket)
        telemetry.round(i,[0,1,2],[5,8],[i,i,i],[1,1,1],[0,0,0],3,3,3)
    result = telemetry.finish({})
    assert result['active_prompt_group_rounds'] == 8
    assert [row['verification_batch'] for row in result['verification_rounds']] == [1,3]
    assert result['generation_profile']['draft_proposal_time_ms'] > 0
    assert result['generation_profile']['spark_correction_update_time_ms'] == 0
    assert result['generation_profile']['target_verification_time_ms'] is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason='real tiny speculative generation requires CUDA')
@pytest.mark.parametrize('method',['fastgrpo','opd_reflex'])
def test_early_eos_and_independent_caps_do_not_terminate_other_responses(monkeypatch, method):
    from test_fastgrpo_rewrite import tiny, run
    if method == 'fastgrpo':
        from helper import fastgrpo_generate as engine
        name = 'sampling'
    else:
        from helper import opd_generate as engine
        name = 'sample_target_with_metadata'
    original = getattr(engine,name)

    def inject_eos(*args,**kwargs):
        result = original(*args,**kwargs)
        if args[0].shape[1] == 1:
            tokens = result if method=='fastgrpo' else result[0]
            tokens[0,0] = 96
            tokens[1,0] = 3
        return result

    def generate(model, ids, mask, tokenizer, **kwargs):
        return speculative_generate(model, ids, mask, tokenizer, **kwargs)

    monkeypatch.setattr(engine,name,inject_eos)
    model = tiny()
    if method == 'opd_reflex':
        model.enable_opd()
    options = dict(method=method, max_response_tokens=7, opd_b_sharing_scope='per_prompt_group')
    clean, rng, counts, _ = run(generate, model, generation_profile='clean', **options)
    detailed, other_rng, other_counts, _ = run(generate,model,generation_profile='detailed',round_log_frequency=1,**options)
    assert clean['generated_token_ids'] == detailed['generated_token_ids']
    assert torch.equal(rng,other_rng) and counts == other_counts
    ids = detailed['generated_token_ids']
    assert ids[:2] == [[96],[96]]
    assert max(map(len,ids[2:])) == 7 and all(2 <= len(row) <= 7 for row in ids[2:])
    assert detailed['response_verification_rounds'][:2] == [0,0]
    assert detailed['prompt_token_lengths'] == [2,3]
    assert [len(row) for row in detailed['all_draft_input_ids']] == [1+len(ids[0]),1+len(ids[1]),
        2+len(ids[2]),2+len(ids[3])]
    assert detailed['generation_memory']['correction_matrix_count'] == (2 if method=='opd_reflex' else 0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='actual GRPO optimizer requires CUDA')
def test_actual_updates_are_logged_even_when_logical_grpo_label_does_not_change(tmp_path):
    command = [sys.executable, str(ROOT/'tests/tiny_training_runner.py'), str(tmp_path), 'fastgrpo', 'run',
        '--rebuttal_experiment','smoke','--rebuttal_model_key','tiny','--generation_profile','clean',
        '--grpo_iteration_num','2','--max_target_optimizer_steps','2','--eval_interval_updates','1',
        '--max_response_tokens','9','--max_training_token','16','--logps_chunk_size','4']
    result = subprocess.run(command, capture_output=True, text=True, env=dict(os.environ,TQDM_DISABLE='1'),timeout=120)
    assert result.returncode == 0, result.stderr[-6000:]
    run = tmp_path/'run'
    with (run/'logs/optimizer_updates.csv').open() as stream:
        updates = list(csv.DictReader(stream))
    assert [int(row['target_optimizer_steps']) for row in updates] == [1,2]
    assert [int(row['rollouts_since_update']) for row in updates] == [1,0]
    assert [int(row['prompts_since_update']) for row in updates] == [2,0]
    actual = json.loads((run/'summary.json').read_text())
    assert actual['target_optimizer_steps'] == 2
    assert actual['actual_draft_optimizer_steps'] == 1
    assert actual['actual_rollouts'] == 1
    assert actual['training_wall_time_s'] >= float(updates[-1]['cumulative_training_wall_time_s'])
    snapshots = run/'checkpoints/evaluation'
    assert (snapshots/'initial_updates_0000000/snapshot.json').is_file()
    assert (snapshots/'intermediate_updates_0000001/snapshot.json').is_file()
    assert (snapshots/'final_updates_0000002/snapshot.json').is_file()


def test_independent_seed_sample_std_and_paired_matched_update_speedups():
    def make(method,seed,updates,duration):
        row = dict(cumulative_training_wall_time_s=duration, reward_mean=.5)
        return SimpleNamespace(method=method,seed=seed,path=f'{method}-{seed}',evaluations=[],
            eval_signature=None,updates={index:row for index in updates}, final_wall=duration,
            metrics_at=lambda index: dict.fromkeys(('reward','correctness','accuracy','response_length',
                'generated_tokens','aal','acceptance_rate','generation_tokens_s','generation_time_s',
                'training_wall_time_s'), duration))
    runs = [make('fastgrpo',42,[2,3],6.),make('opd_reflex',42,[2,3,4],3.),
            make('fastgrpo',123,[1,2],12.),make('opd_reflex',123,[2,3],4.)]
    results = comparison(runs)
    assert results['matched_target_optimizer_steps'] == 2
    assert results['common_observed_budget_s'] == 3
    speed = next(row for row in results['summary'] if row['method']=='spark_over_fastgrpo')
    assert speed['mean'] == 2.5 and speed['n'] == 2
    assert speed['sample_std'] == pytest.approx(2**-.5)
    assert stat([.9])['sample_std'] is None


def test_legacy_cadence_indexes_actual_updates_without_fabricating_wall_or_reward(tmp_path):
    from scripts.summarize_rebuttal import legacy_updates

    path = tmp_path/'rollouts.csv'
    path.write_text('total_prompts,iter_generation_time_s,cumulative_generation_time_s,'
                    'cumulative_rollout_tokens,cumulative_aal,cumulative_acceptance_rate\n'
                    '8,1,1,100,1.2,0.1\n8,0,1,100,1.2,0.1\n8,2,3,180,1.3,0.2\n')
    config = dict(args=dict(repeated_generate_nums=8))
    rows = legacy_updates(tmp_path, {'legacy_rollout_timing':path.name, 'comparable_generation_time':True}, config,
                          {'optimizer_step_cadence':[[8,1],[16,3]]})
    assert [row['target_optimizer_steps'] for row in rows] == [1,3]
    assert rows[-1]['cumulative_responses'] == 128
    assert rows[-1]['cumulative_generation_tokens_s'] == 60
    assert rows[-1]['cumulative_training_wall_time_s'] is None
    assert 'reward_mean' not in rows[-1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason='frozen engines require CUDA')
def test_frozen_comparison_uses_both_engines_and_saved_projector_on_tiny_data(tmp_path, monkeypatch):
    from scripts import benchmark_rebuttal
    subprocess.run([sys.executable, str(ROOT/'tests/tiny_training_runner.py'), str(tmp_path)],
                   check=True, capture_output=True, timeout=90)
    projector = torch.arange(256, dtype=torch.float32).reshape(32,8)/256
    projector_file = tmp_path/'learned_projector_fixture.pt'
    torch.save(projector,projector_file)
    monkeypatch.setattr(benchmark_rebuttal,'configurations',lambda args:[(2,2,7)])
    monkeypatch.setenv('OPD_PROPOSAL_MODE','dense')
    output = tmp_path/'benchmark'
    benchmark_rebuttal.main(['--target-model',str(tmp_path/'model'),'--model-key','tiny',
        '--draft-checkpoint',str(tmp_path/'draft.pth'),'--projector-checkpoint',str(projector_file),
        '--dataset-path',str(tmp_path/'train.json'),'--dataset','synthetic','--output',str(output),
        '--sweep','long_context','--max-prompt-length','32','--verification-capacity','28',
        '--max-verification-num','7','--max-draft-k','2','--max-draft-token-length','3',
        '--repetitions','2','--round-log-frequency','1'])
    report = json.loads((output/'summary.json').read_text())
    assert [row['method'] for row in report['results']] == ['fastgrpo','opd_reflex']
    assert report['paired_speedups'][0]['repetitions'] == 2
    assert report['results'][0]['correction_matrix_count_mean'] == 0
    assert report['results'][1]['correction_matrix_count_mean'] == 2
    assert report['diagnostics'][1]['spark_correction_application_time_ms_mean'] is None
    assert report['diagnostics'][1]['spark_correction_update_time_ms_mean'] > 0


def test_random_rebuttal_ignores_draft_checkpoint_and_starts_fresh(tmp_path):
    script = ROOT / 'scripts/rebuttal/train_qwen25_7b_gsm8k_opd_reflex_seed123.sh'
    env = dict(os.environ, DRY_RUN='true', DRAFT_INITIALIZATION_MODE='random',
               DRAFT_CHECKPOINT=str(tmp_path/'missing'), OUTPUT_ROOT=str(tmp_path/'runs'))
    result = subprocess.run(['bash',str(script)], env=env, text=True, capture_output=True, check=True)
    argv = shlex.split(next(line for line in result.stdout.splitlines() if line.startswith('Command')).split(':',1)[1])
    assert argv[argv.index('--method')+1] == 'opd_reflex'
    assert argv[argv.index('--draft_initialization_mode')+1] == 'random'
    assert argv[argv.index('--adapter_path')+1] == ''
    assert argv[argv.index('--resume_checkpoint')+1] == ''
    assert not (tmp_path/'runs').exists()
    inherited_resume = subprocess.run(['bash',str(script)], env=dict(env, RESUME='auto'),
                                      text=True, capture_output=True)
    assert inherited_resume.returncode == 2
    assert 'fresh run' in inherited_resume.stderr


def test_missing_pretrained_draft_preflight_identifies_initialization(tmp_path):
    model = tmp_path/'model'
    model.mkdir()
    (model/'config.json').write_text('{}')
    data = tmp_path/'data.json'
    data.write_text('[]')
    script = ROOT / 'scripts/rebuttal/train_qwen25_7b_gsm8k_opd_reflex_seed123.sh'
    env = dict(os.environ, DRY_RUN='false', PYTHON_BIN=sys.executable, MODEL=str(model), DATASET_PATH=str(data),
               EVAL_DATASET_PATH=str(data), DRAFT_CHECKPOINT=str(tmp_path/'missing'),
               DRAFT_INITIALIZATION_MODE='pretrained', OUTPUT_ROOT=str(tmp_path/'runs'))
    result = subprocess.run(['bash',str(script)], env=env, text=True, capture_output=True)
    assert result.returncode == 2
    assert 'draft checkpoint not found' in result.stderr
    assert 'DRAFT_INITIALIZATION_MODE=random' in result.stderr
    assert not (tmp_path/'runs').exists()


def test_random_init_signature_cannot_pool_with_pretrained():
    from scripts.summarize_rebuttal import BASE_ARGS, signature

    config = dict(args=dict.fromkeys(BASE_ARGS), inputs={key:{'sha256':key}
                  for key in ('target','train_dataset')})
    config['inputs'].update(draft=None, target_adapter=None)
    config['dataset'] = dict(selected_rows_sha256='rows', selected_sample_count=2, prompt_order_protocol='shuffle')
    config['environment'] = dict(gpu='B200', gpu_memory_bytes=1, world_size=1, gpu_count=1,
        torch_version='version', cuda_build='build', driver='driver',
        software=dict.fromkeys(('transformers','peft','triton','math-verify')))
    config['args']['draft_initialization_mode'] = 'random'
    random_settings, _ = signature(config, 'fastgrpo')
    assert random_settings['assets']['draft'] is None
    config['args']['draft_initialization_mode'] = 'pretrained'
    config['inputs']['draft'] = dict(sha256='pretrained-draft')
    pretrained, _ = signature(config, 'fastgrpo')
    assert random_settings != pretrained
