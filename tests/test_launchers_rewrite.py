import ast
import json
import os
from pathlib import Path
import shlex
import subprocess
import pytest

ROOT=Path(__file__).resolve().parents[1]
MODELS=['qwen25_1p5b','qwen25_3b','qwen25_7b','qwen25_14b','qwen3_1p7b','qwen3_4b']

def command(name):
    env={**os.environ,'DRY_RUN':'true','RUN_NAME':'paired-check','RUN_DIR':'/tmp/specnaacl-paired-check'}
    out=subprocess.check_output(['bash',str(ROOT/name)],env=env,text=True)
    line=next(line for line in out.splitlines() if line.startswith('Command'))
    return shlex.split(line.split(':',1)[1])

def flags(tokens):
    start=tokens.index(next(x for x in tokens if x.endswith('grpo_speculative.py') or x.endswith('train_draft.py')))+1
    return dict(zip(tokens[start::2],tokens[start+1::2]))

@pytest.mark.parametrize('model',MODELS)
def test_paired_training_scripts_share_checkpoint_and_all_non_opd_config(model):
    opd=flags(command(f'train_{model}.sh'));baseline=flags(command(f'train_{model}_fastgrpo.sh'))
    assert baseline.pop('--method')=='fastgrpo';assert opd.pop('--method')=='opd_reflex'
    assert not any('opd_' in k for k in baseline)
    opd={k:v for k,v in opd.items() if 'opd_' not in k and k!='--kv_gather_strategy'}
    assert baseline==opd
    assert baseline['--train_option']=='simplelr_abel_level3to5'
    assert baseline['--target_lr']=='1e-6';assert baseline['--draft_lr']=='1e-4'
    assert baseline['--draft_accumulation_steps']=='1'
    assert baseline['--adapter_path'].endswith(f'/pretrain/{model}/latest_checkpoint')
    assert not any('vocab_mapping' in k or 'draft_config' in k for k in baseline)

@pytest.mark.parametrize('model',MODELS)
def test_pretrain_defaults_paths(model):
    c=flags(command(f'pretrain_{model}.sh'))
    assert c['--num_epochs']=='5'
    assert c['--dataset_dir'].endswith('/sharegpt/ShareGPT_V4.3_unfiltered_cleaned_split.json')
    assert c['--saved_model_dir']=='/tmp/specnaacl-paired-check/checkpoints'
    assert c['--log_dir']=='/tmp/specnaacl-paired-check/logs'


def test_source_manifest_is_byte_identical_to_original_checkout():
    import hashlib
    manifest=json.loads((ROOT/'sources/FastGRPO/SOURCE_MANIFEST.json').read_text())
    for name,expected in manifest['sha256'].items():
        assert hashlib.sha256((ROOT/'sources/FastGRPO'/name).read_bytes()).hexdigest()==expected
        assert hashlib.sha256((ROOT.parent/'FastGRPO-main'/name).read_bytes()).hexdigest()==expected
    assert (ROOT/'helper/modeling_draft.py').read_bytes()==(ROOT/'sources/FastGRPO/helper/modeling_draft.py').read_bytes()


def test_local_simplelr_parquet_prompt_arrays(tmp_path):
    import pandas as pd
    from helper.get_QAs import get_QAs_from_path
    path=tmp_path/'train.parquet'
    pd.DataFrame([dict(prompt=[dict(role='user',content='2+2?')],reward_model=dict(ground_truth='4'))]).to_parquet(path)
    assert get_QAs_from_path(path,'train')==[dict(question='2+2?',answer='\\boxed{4}')]


@pytest.mark.parametrize('stopped',[True,False])
def test_auto_resume_resumes_bounded_run_but_not_completed_run(tmp_path,stopped):
    root=tmp_path/'train'
    active=root/'qwen25_3b__simplelr__method-fastgrpo__test'
    (active/'checkpoints/resume').mkdir(parents=True)
    checkpoint=active/'checkpoints/resume/latest.pt';checkpoint.touch()
    (active/'summary.json').write_text(json.dumps({'stopped_by_max_grpo_steps':stopped}))
    (root/'active_run').symlink_to(active)
    env={k:v for k,v in os.environ.items() if k not in ('RUN_DIR','RUN_NAME')}
    env.update(DRY_RUN='true',RESUME='auto',TRAIN_MODEL_ROOT=str(root))
    output=subprocess.check_output(['bash',str(ROOT/'train_qwen25_3b_fastgrpo.sh')],env=env,text=True)
    line=next(line for line in output.splitlines() if line.startswith('Command'))
    parsed=flags(shlex.split(line.split(':',1)[1]))
    assert parsed['--resume_checkpoint']==(str(checkpoint) if stopped else '')
