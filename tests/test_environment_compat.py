"""Installed-stack admission and actual Transformers decoder/cache compatibility."""
from copy import deepcopy
from types import SimpleNamespace
import weakref

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from scripts import validate_environment as validator
from helper.transformers_compat import DynamicCache, LegacyDynamicCache, prepare_target_decoder_api
from helper.environment_checks import probe_training_runtime

USER_STACK = {
    'torch':'2.13.0+cu130', 'transformers':'5.12.1', 'peft':'0.21.1',
    'datasets':'5.0.1', 'accelerate':'1.15.0', 'safetensors':'0.8.0',
    'numpy':'2.3.5', 'pandas':'3.0.6', 'tqdm':'4.70.1', 'math-verify':'0.9.0',
    'latex2sympy2-extended':'1.11.0', 'triton':'3.7.1', 'matplotlib':'3.11.2',
    'packaging':'26.3',
}
PINS=dict(validator.pinned_requirements(validator.REPO_ROOT/'requirements.txt'))


@pytest.mark.parametrize('name,actual',USER_STACK.items())
def test_all_reported_user_versions_are_admitted_after_api_checks(name,actual):
    assert validator.version_matches(name,PINS[name],actual)
    assert not validator.version_matches(name,PINS[name],actual,strict=True)


@pytest.mark.parametrize('name,value',[('torch','3.0.0'),('transformers','6.0.0'),
    ('transformers','4.50.0'),('peft','0.22.0'),('triton','4.0.0'),('numpy','3.0.0')])
def test_unhandled_api_families_are_not_blindly_accepted(name,value):
    assert not validator.version_matches(name,PINS[name],value)


def test_python_only_does_not_read_requirements_or_import_runtime(monkeypatch):
    def forbidden(*args):pytest.fail('python-only attempted to load dependencies')
    monkeypatch.setattr(validator,'version',forbidden)
    monkeypatch.setattr(validator.importlib,'import_module',forbidden)
    validator.main(['--python-only','--requirements','missing.txt'])


def test_missing_peft_api_remains_an_error(monkeypatch):
    monkeypatch.setattr(validator,'pinned_requirements',lambda _:iter([('peft','0.17.1')]))
    monkeypatch.setattr(validator,'version',lambda _:USER_STACK['peft'])
    monkeypatch.setattr(validator.importlib,'import_module',lambda _:SimpleNamespace())
    with pytest.raises(RuntimeError,match='required APIs missing'):
        validator.main([])


def test_decoder_adapter_preserves_native_forward_values_and_weight_names():
    torch.manual_seed(42)
    config=Qwen2Config(vocab_size=97,hidden_size=32,intermediate_size=64,
        num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,
        attention_dropout=0.,torch_dtype=torch.float32)
    config._attn_implementation='sdpa'
    native=Qwen2ForCausalLM(config).eval()
    adapted=deepcopy(native);before=list(adapted.state_dict())
    prepare_target_decoder_api(adapted)
    prepare_target_decoder_api(adapted)  # idempotent even after repeated setup
    ids=torch.tensor([[2,4,7,8]])
    with torch.no_grad():
        a=native(ids,use_cache=False).logits;b=adapted(ids,use_cache=False).logits
    assert torch.equal(a,b)
    assert before==list(adapted.state_dict())


def test_modern_cache_legacy_lists_follow_update_crop_repeat_and_indexed_writes():
    cache=DynamicCache()
    keys=torch.randn(1,2,5,8);values=torch.randn_like(keys)
    cache.update(keys,values,0)
    cache.update(keys+1,values+1,1)
    assert len(cache.key_cache)==2
    assert torch.equal(torch.stack(cache.key_cache),torch.stack([keys,keys+1]))
    cache.batch_repeat_interleave(2);cache.crop(3)
    replacement=torch.full((1,2,3,8),.3)
    cache.key_cache[0]=replacement
    assert torch.equal(cache[0][0],replacement)
    assert cache.value_cache[0].shape==(2,2,3,8)
    if isinstance(cache,LegacyDynamicCache):
        assert cache.layers[0].keys is replacement
        reference=weakref.ref(cache)
        del cache
        assert reference() is None  # no cycle retaining GPU KV between rollouts


def test_cpu_runtime_probe_executes_pretrain_lora_and_checkpoint_roundtrip():
    rng=torch.get_rng_state()
    assert 'checkpoint' in probe_training_runtime('cpu')
    assert torch.equal(rng,torch.get_rng_state())


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA API probe')
def test_cuda_runtime_probe_executes_the_actual_sdpa_and_optimizer():
    rng=torch.cuda.get_rng_state()
    assert 'pretrain' in probe_training_runtime('cuda')
    assert torch.equal(rng,torch.cuda.get_rng_state())
