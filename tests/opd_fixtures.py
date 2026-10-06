"""Deterministic tiny fixtures, REAL sampler/tree/verifier/OPD, not throughput data."""
import ast
from pathlib import Path
from types import SimpleNamespace
from copy import deepcopy
import math
import time
import torch
from helper.opd_reflex import OPDReflex
from helper.method_config import resolve_method
from helper.rollout_history import RolloutHistory
from helper.tree_verification import pack_tree,trace_verified_path
from helper.sampling import build_sampling_probs,sample_from_probs,sample_target_from_logits

class Cache:
    def __init__(self):
        self.layers = []

    def get_seq_length(self):
        return self.layers[0].keys.shape[-2] if self.layers else 0

    def crop(self, length):
        for layer in self.layers:
            layer.keys = layer.keys[..., :length, :]
            layer.values = layer.values[..., :length, :]

    def batch_repeat_interleave(self, repeats):
        for layer in self.layers:
            layer.keys = layer.keys.repeat_interleave(repeats, 0)
            layer.values = layer.values.repeat_interleave(repeats, 0)

class TinyModel:
    is_eagle3_specforge = True
    device, dtype, compact_vocab_size = torch.device("cpu"), torch.bfloat16, 17

    def __init__(self):
        generator = torch.Generator().manual_seed(121)
        self.embedding = torch.randn(17, 8, generator=generator).bfloat16()
        self.target_head = torch.nn.Linear(8, 17, bias=False).bfloat16()
        self.draft_head = torch.nn.Linear(8, 17, bias=False).bfloat16()
        with torch.no_grad():
            self.target_head.weight.copy_(torch.randn(17, 8, generator=generator) * .3)
            self.draft_head.weight.copy_(torch.randn(17, 8, generator=generator) * .3)
        self.calls, self.masks = 0, []
        self.target_model = SimpleNamespace(device=self.device, dtype=self.dtype,
                                           model=self.target_forward, lm_head=self.target_head)

    def target_forward(self, input_ids, attention_mask, past_key_values, **kwargs):
        self.calls += 1
        self.masks.append(attention_mask.clone())
        hidden = self.embedding[input_ids]
        keys = hidden.unsqueeze(1)
        if past_key_values.layers:
            keys = torch.cat((past_key_values.layers[0].keys, keys), -2)
        visible = (attention_mask[:, 0] == 0).to(self.dtype)
        # Attention depends on ancestry/history, not just current token.
        hidden = hidden + visible.matmul(keys[:, 0]) / visible.sum(-1, keepdim=True).clamp_min(1)
        past_key_values.layers = [SimpleNamespace(keys=keys, values=keys.clone())]
        return SimpleNamespace(last_hidden_state=hidden, past_key_values=past_key_values)

    def __call__(self, hidden_states, input_ids, past_key_values=None, **kwargs):
        hidden = (hidden_states + self.embedding[input_ids]) * .5
        keys = hidden.unsqueeze(1)
        if past_key_values is not None:
            keys = torch.cat((past_key_values[0][0], keys), -2)
        return dict(hidden_states=hidden, next_feature_states=hidden,
                    past_key_values=[(keys, keys.clone())])

    def compute_compact_logits(self, hidden):
        return self.draft_head(hidden)

    def compact_to_target_ids(self, device):
        return torch.arange(17, device=device)

class CountModel(TinyModel):
    def __init__(self,device='cpu'):
        super().__init__()
        self.device=torch.device(device);self.target_model.device=self.device
        self.embedding=self.embedding.to(device);self.target_head.to(device);self.draft_head.to(device)
        self.mapping=torch.arange(17,device=device);self.draft_calls=0
    def __call__(self,*args,**kwargs):
        self.draft_calls+=1
        return super().__call__(*args,**kwargs)
    def compact_to_target_ids(self,device=None):return self.mapping

def load_rollout(device='cpu',history_type=RolloutHistory):
    path=Path(__file__).resolve().parents[1]/'helper/specualtive_generate.py'
    tree=ast.parse(path.read_text())
    for n in ast.walk(tree):
        if isinstance(n,ast.FunctionDef) and n.name=='get_attention_mask':
            n.args.defaults[1]=ast.Constant(device)
    fns=[n for n in tree.body if isinstance(n,ast.FunctionDef)]
    scope=dict(torch=torch,time=time,math=math,deepcopy=deepcopy,DynamicCache=Cache,
               OPDReflex=OPDReflex,resolve_method=resolve_method,RolloutHistory=history_type,
               pack_tree=pack_tree,trace_verified_path=trace_verified_path,
               build_sampling_probs=build_sampling_probs,sample_from_probs=sample_from_probs,
               sample_target_from_logits=sample_target_from_logits)
    exec(compile(ast.fix_missing_locations(ast.Module(body=fns,type_ignores=[])),str(path),'exec'),scope)
    generate=scope['speculative_generate'];generate._test_scope=scope
    return generate
